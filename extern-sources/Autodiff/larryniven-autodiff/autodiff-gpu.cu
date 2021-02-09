#include "autodiff/autodiff-gpu.h"
#include "autodiff/autodiff-op-gpu.h"
#include "autodiff/autodiff-op.h"
#include "la/la-gpu.h"
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <algorithm>

namespace autodiff {

    namespace gpu {

        void zeros_eval(std::shared_ptr<op_t> t)
        {
            std::vector<unsigned int> const& sizes
                = *std::static_pointer_cast<std::vector<unsigned int>>(t->data);

            la::gpu::tensor<double> z;
            z.resize(sizes);

            t->output = std::make_shared<la::gpu::tensor<double>>(z);
        }

        void zeros_grad(std::shared_ptr<op_t> t)
        {}

        void weak_var_eval(std::shared_ptr<op_t> t)
        {
            int shift;
            std::vector<unsigned int> sizes;

            std::tie(shift, sizes) = *std::static_pointer_cast<
                std::pair<int, std::vector<unsigned int>>>(t->data);

            auto ch = get_child(t, 0);

            auto& v = get_output<la::gpu::tensor_like<double>>(ch);
            la::gpu::weak_tensor<double> w_v { v.data() + shift, sizes };
            t->output = std::make_shared<la::gpu::weak_tensor<double>>(w_v);

            if (ch->grad_needed && ch->grad == nullptr) {
                la::gpu::tensor<double> g;
                g.resize(v.sizes());
                ch->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            if (ch->grad_needed) {
                auto& z = get_grad<la::gpu::tensor_like<double>>(ch);
                la::gpu::weak_tensor<double> w_z { z.data() + shift, sizes };
                t->grad = std::make_shared<la::gpu::weak_tensor<double>>(w_z);
            }
        }

        void weak_var_grad(std::shared_ptr<op_t> t)
        {
        }

        struct subtensor_op {

            double *sub_data;
            double const *orig_data;
            unsigned int dim;
            unsigned int const *shift;
            unsigned int const *sizes;
            unsigned int const *a_sizes;

            __device__
            void index_to_coord(int index, double *coord)
            {
                int i = index;

                for (int d = dim - 1; d > 0; --d) {
                    int c = i % sizes[d];
                    coord[d] = c;
                    i = (i - c) / sizes[d];
                }

                coord[0] = i;
            }

            __device__
            int coord_to_index(double *coord)
            {
                int result = 0;

                for (int d = 0; d < dim; ++d) {
                    result *= a_sizes[d];
                    result += coord[d];
                }

                return result;
            }

            __device__
            void operator()(int i)
            {
                double *coord = (double*) malloc(dim);

                for (int i = 0; i < dim; ++i) {
                    coord[i] = 0;
                }

                index_to_coord(i, coord);

                for (int j = 0; j < dim; ++j) {
                    coord[j] += shift[j];
                }

                int j = coord_to_index(coord);

                sub_data[i] += orig_data[j];

                free(coord);
            }

        };

        void subtensor_eval(std::shared_ptr<op_t> t)
        {
            auto& orig = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));

            std::vector<unsigned int> shift;
            std::vector<unsigned int> sizes;

            std::tie(shift, sizes) = *std::static_pointer_cast<std::pair<std::vector<unsigned int>,
                std::vector<unsigned int>>>(t->data);

            assert(orig.dim() == shift.size());

            for (int i = 0; i < shift.size(); ++i) {
                assert(shift[i] + sizes[i] <= orig.size(i));
            }

            if (t->output == nullptr) {
                la::gpu::tensor<double> c;
                c.resize(sizes);
                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(c));
            }

            auto& sub = get_output<la::gpu::tensor_like<double>>(t);

            thrust::counting_iterator<int> sub_begin {0};
            thrust::counting_iterator<int> sub_end = sub_begin + sub.vec_size();

            la::gpu::vector<unsigned int> dshift { la::cpu::vector<unsigned int>(shift) };
            la::gpu::vector<unsigned int> dsizes { la::cpu::vector<unsigned int>(sizes) };
            la::gpu::vector<unsigned int> dorigsizes { la::cpu::vector<unsigned int>(orig.sizes()) };

            thrust::for_each(sub_begin, sub_end,
                subtensor_op { sub.data(), orig.data(), dshift.size(),
                    dshift.data(), dsizes.data(), dorigsizes.data() });
        }

        struct subtensor_grad_op {

            double *orig_grad_data;
            double const *sub_grad_data;
            unsigned int dim;
            unsigned int const *shift;
            unsigned int const *sizes;
            unsigned int const *a_sizes;

            __device__
            void index_to_coord(int index, double *coord)
            {
                int i = index;

                for (int d = dim - 1; d > 0; --d) {
                    int c = i % sizes[d];
                    coord[d] = c;
                    i = (i - c) / sizes[d];
                }

                coord[0] = i;
            }

            __device__
            int coord_to_index(double *coord)
            {
                int result = 0;

                for (int d = 0; d < dim; ++d) {
                    result *= a_sizes[d];
                    result += coord[d];
                }

                return result;
            }

            __device__
            void operator()(int i)
            {
                double *coord = (double*) malloc(dim);

                for (int i = 0; i < dim; ++i) {
                    coord[i] = 0;
                }

                index_to_coord(i, coord);

                for (int j = 0; j < dim; ++j) {
                    coord[j] += shift[j];
                }

                int j = coord_to_index(coord);

                orig_grad_data[j] += sub_grad_data[i];

                free(coord);
            }

        };

        void subtensor_grad(std::shared_ptr<op_t> t)
        {
            auto ch = get_child(t, 0);

            auto& orig = get_output<la::gpu::tensor_like<double>>(ch);

            std::vector<unsigned int> shift;
            std::vector<unsigned int> sizes;

            std::tie(shift, sizes) = *std::static_pointer_cast<std::pair<std::vector<unsigned int>,
                std::vector<unsigned int>>>(t->data);

            if (ch->grad == nullptr) {
                la::gpu::tensor<double> c;
                c.resize(orig.sizes());
                ch->grad = std::make_shared<la::gpu::tensor<double>>(std::move(c));
            }

            auto& orig_grad = get_grad<la::gpu::tensor_like<double>>(ch);
            auto& sub_grad = get_grad<la::gpu::tensor_like<double>>(t);

            thrust::counting_iterator<int> sub_begin {0};
            thrust::counting_iterator<int> sub_end = sub_begin + sub_grad.vec_size();

            la::gpu::vector<unsigned int> dshift { la::cpu::vector<unsigned int>(shift) };
            la::gpu::vector<unsigned int> dsizes { la::cpu::vector<unsigned int>(sizes) };
            la::gpu::vector<unsigned int> dorigsizes { la::cpu::vector<unsigned int>(orig.sizes()) };

            thrust::for_each(sub_begin, sub_end,
                subtensor_grad_op { orig_grad.data(), sub_grad.data(), dshift.size(),
                    dshift.data(), dsizes.data(), dorigsizes.data() });
        }

        struct split_block_op {

            double *z_data;
            double const *c_data;

            unsigned int last_dim;
            unsigned int c_rows;
            unsigned int c_cols;

            __device__
            void operator()(int k)
            {
                int i = k / c_cols;
                int j = k % c_cols;

                z_data[(j / last_dim) * last_dim * c_rows + i * last_dim + j % last_dim]
                    = c_data[k];
            }

        };

        void split_block_eval(std::shared_ptr<op_t> t)
        {
            int block = *std::static_pointer_cast<int>(t->data);

            auto c = get_child(t, 0);
            auto& ct = get_output<la::gpu::tensor_like<double>>(c);

            if (t->output == nullptr) {
                la::gpu::tensor<double> z;

                std::vector<unsigned int> sizes;
                sizes.push_back(block);

                std::vector<unsigned int> const& ct_sizes = ct.sizes();
                assert(ct_sizes.size() >= 2);
                assert(ct_sizes.back() % block == 0);

                sizes.insert(sizes.end(), ct_sizes.begin(), ct_sizes.end() - 1);
                sizes.push_back(ct_sizes.back() / block);

                z.resize(sizes);

                t->output = std::make_shared<la::gpu::tensor<double>>(z);
            }

            auto& z = get_output<la::gpu::tensor_like<double>>(t);

            auto& c_mat = ct.as_matrix();

            thrust::counting_iterator<int> c_begin { 0 };
            thrust::counting_iterator<int> c_end = c_begin + ct.vec_size();

            thrust::for_each(c_begin, c_end, split_block_op {
                z.data(), ct.data(), c_mat.cols() / block, c_mat.rows(), c_mat.cols()
            });
        }

        struct split_block_grad_op {

            double const *t_grad_data;
            double *c_grad_data;

            unsigned int last_dim;
            unsigned int c_rows;
            unsigned int c_cols;

            __device__
            void operator()(int k)
            {
                int i = k / c_cols;
                int j = k % c_cols;

                c_grad_data[k] += t_grad_data[(j / last_dim) * last_dim * c_rows
                    + i * last_dim + j % last_dim];
            }

        };

        void split_block_grad(std::shared_ptr<op_t> t)
        {
            int block = *std::static_pointer_cast<int>(t->data);

            auto c = get_child(t, 0);
            auto& ct = get_output<la::gpu::tensor_like<double>>(c);

            if (c->grad_needed && c->grad == nullptr) {
                la::gpu::tensor<double> z;

                la::gpu::resize_as(z, ct);

                c->grad = std::make_shared<la::gpu::tensor<double>>(z);
            }

            if (c->grad_needed) {
                auto& t_grad = get_grad<la::gpu::tensor_like<double>>(t);
                auto& c_grad = get_grad<la::gpu::tensor_like<double>>(c);

                auto& c_mat = ct.as_matrix();

                thrust::counting_iterator<int> c_begin { 0 };
                thrust::counting_iterator<int> c_end = c_begin + ct.vec_size();

                thrust::for_each(c_begin, c_end, split_block_grad_op {
                    t_grad.data(), c_grad.data(), c_mat.cols() / block, c_mat.rows(), c_mat.cols()
                });
            }
        }

        void mul_eval(std::shared_ptr<op_t> t)
        {
            auto& a = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));
            auto& b = get_output<la::gpu::tensor_like<double>>(get_child(t, 1));

            if (t->output == nullptr) {
                la::gpu::tensor<double> c;

                std::vector<unsigned int> sizes = a.sizes();
                sizes.pop_back();
                sizes.push_back(b.size(b.dim() - 1));

                c.resize(sizes);

                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(c));
            }

            auto& c = get_output<la::gpu::tensor_like<double>>(t);

            // la::cpu::tensor<double> hc = la::gpu::to_host(c);
            // la::cpu::tensor<double> ha = la::gpu::to_host(a);
            // la::cpu::tensor<double> hb = la::gpu::to_host(b);

            // la::cpu::mul(hc, ha, hb);

            // la::gpu::to_device(c, hc);

            la::gpu::mul(c, a, b);
        }

        void mul_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::tensor_like<double>>(t);

            auto a_o = get_child(t, 0);
            auto b_o = get_child(t, 1);

            auto& a = get_output<la::gpu::tensor_like<double>>(a_o);
            auto& b = get_output<la::gpu::tensor_like<double>>(b_o);

            if (a_o->grad_needed && a_o->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, a);
                a_o->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            if (b_o->grad_needed && b_o->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, b);
                b_o->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& a_grad = get_grad<la::gpu::tensor_like<double>>(a_o);
            auto& b_grad = get_grad<la::gpu::tensor_like<double>>(b_o);

            if (a_o->grad_needed) {
                // la::cpu::tensor<double> ha_grad = la::gpu::to_host(a_grad);
                // la::cpu::tensor<double> hgrad = la::gpu::to_host(grad);
                // la::cpu::tensor<double> hb = la::gpu::to_host(b);

                // la::cpu::rtmul(ha_grad, hgrad, hb);

                // la::gpu::to_device(a_grad, ha_grad);

                la::gpu::rtmul(a_grad, grad, b);
            }

            if (b_o->grad_needed) {
                // la::cpu::tensor<double> hb_grad = la::gpu::to_host(b_grad);
                // la::cpu::tensor<double> hgrad = la::gpu::to_host(grad);
                // la::cpu::tensor<double> ha = la::gpu::to_host(a);

                // la::cpu::ltmul(hb_grad, ha, hgrad);

                // la::gpu::to_device(b_grad, hb_grad);

                la::gpu::ltmul(b_grad, a, grad);
            }
        }

        void emul_eval(std::shared_ptr<op_t> t)
        {
            auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 1));

            if (t->output == nullptr) {
                la::gpu::tensor<double> z;
                la::gpu::resize_as(z, u);
                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(z));
            }

            auto& z = get_output<la::gpu::tensor_like<double>>(t);

            // la::cpu::tensor<double> hz = la::gpu::to_host(z);
            // la::cpu::tensor<double> hu = la::gpu::to_host(u);
            // la::cpu::tensor<double> hv = la::gpu::to_host(v);

            // la::cpu::emul(hz, hu, hv);

            // la::gpu::to_device(z, hz);

            la::gpu::emul(z, u, v);
        }

        void emul_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::tensor_like<double>>(t);

            auto u_o = get_child(t, 0);
            auto v_o = get_child(t, 1);

            auto& u = get_output<la::gpu::tensor_like<double>>(u_o);
            auto& v = get_output<la::gpu::tensor_like<double>>(v_o);

            if (u_o->grad_needed && u_o->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, u);
                u_o->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            if (v_o->grad_needed && v_o->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, v);
                v_o->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& u_grad = get_grad<la::gpu::tensor_like<double>>(u_o);
            auto& v_grad = get_grad<la::gpu::tensor_like<double>>(v_o);

            if (u_o->grad_needed) {
                // la::cpu::tensor<double> hu_grad = la::gpu::to_host(u_grad);
                // la::cpu::tensor<double> hgrad = la::gpu::to_host(grad);
                // la::cpu::tensor<double> hv = la::gpu::to_host(v);

                // la::cpu::emul(hu_grad, hgrad, hv);

                // la::gpu::to_device(u_grad, hu_grad);

                la::gpu::emul(u_grad, grad, v);
            }

            if (v_o->grad_needed) {
                // la::cpu::tensor<double> hv_grad = la::gpu::to_host(v_grad);
                // la::cpu::tensor<double> hgrad = la::gpu::to_host(grad);
                // la::cpu::tensor<double> hu = la::gpu::to_host(u);

                // la::cpu::emul(hv_grad, hgrad, hu);

                // la::gpu::to_device(v_grad, hv_grad);

                la::gpu::emul(v_grad, grad, u);
            }
        }

        void emul_to_eval(std::shared_ptr<op_t> t)
        {
            auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, 1));
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 2));

            auto& z = get_output<la::gpu::tensor_like<double>>(t);

            // la::cpu::tensor<double> hz = la::gpu::to_host(z);
            // la::cpu::tensor<double> hu = la::gpu::to_host(u);
            // la::cpu::tensor<double> hv = la::gpu::to_host(v);

            // la::cpu::emul(hz, hu, hv);

            // la::gpu::to_device(z, hz);

            la::gpu::emul(z, u, v);

            auto storage = get_child(t, 0);

            if (t->grad_needed && storage->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, z);
                storage->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            t->grad = storage->grad;
        }

        void emul_to_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::tensor_like<double>>(t);

            auto u_o = get_child(t, 1);
            auto v_o = get_child(t, 2);

            auto& u = get_output<la::gpu::tensor_like<double>>(u_o);
            auto& v = get_output<la::gpu::tensor_like<double>>(v_o);

            if (u_o->grad_needed && u_o->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, u);
                u_o->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            if (v_o->grad_needed && v_o->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, v);
                v_o->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& u_grad = get_grad<la::gpu::tensor_like<double>>(u_o);
            auto& v_grad = get_grad<la::gpu::tensor_like<double>>(v_o);

            if (u_o->grad_needed) {
                // la::cpu::tensor<double> hu_grad = la::gpu::to_host(u_grad);
                // la::cpu::tensor<double> hgrad = la::gpu::to_host(grad);
                // la::cpu::tensor<double> hv = la::gpu::to_host(v);

                // la::cpu::emul(hu_grad, hgrad, hv);

                // la::gpu::to_device(u_grad, hu_grad);

                la::gpu::emul(u_grad, grad, v);
            }

            if (v_o->grad_needed) {
                // la::cpu::tensor<double> hv_grad = la::gpu::to_host(v_grad);
                // la::cpu::tensor<double> hgrad = la::gpu::to_host(grad);
                // la::cpu::tensor<double> hu = la::gpu::to_host(u);

                // la::cpu::emul(hv_grad, hgrad, hu);

                // la::gpu::to_device(v_grad, hv_grad);

                la::gpu::emul(v_grad, grad, u);
            }
        }

        void logistic_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));

            if (t->output == nullptr) {
                la::gpu::tensor<double> z;
                la::gpu::resize_as(z, v);
                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(z));
            }

            auto& z = get_output<la::gpu::tensor_like<double>>(t);

            // la::cpu::tensor<double> hz = la::gpu::to_host(z);
            // la::cpu::tensor<double> hv = la::gpu::to_host(v);

            // op::logistic(hz, hv);

            // la::gpu::to_device(z, hz);

            op::gpu::logistic(z, v);
        }

        void logistic_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::tensor_like<double>>(t);
            auto& output = get_output<la::gpu::tensor_like<double>>(t);

            auto ch = get_child(t, 0);
            auto& ch_t = get_output<la::gpu::tensor_like<double>>(ch);

            if (ch->grad_needed && ch->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, ch_t);
                ch->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& result = get_grad<la::gpu::tensor_like<double>>(ch);

            if (ch->grad_needed) {
                // la::cpu::tensor<double> hresult = la::gpu::to_host(result);
                // la::cpu::tensor<double> hgrad = la::gpu::to_host(grad);
                // la::cpu::tensor<double> houtput = la::gpu::to_host(output);

                // op::ilogistic_grad(hresult, hgrad, houtput);

                // la::gpu::to_device(result, hresult);

                op::gpu::ilogistic_grad(result, grad, output);
            }
        }

        void tanh_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));

            if (t->output == nullptr) {
                la::gpu::tensor<double> z;
                la::gpu::resize_as(z, v);
                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(z));
            }

            auto& z = get_output<la::gpu::tensor_like<double>>(t);

            // la::cpu::tensor<double> hz = la::gpu::to_host(z);
            // la::cpu::tensor<double> hv = la::gpu::to_host(v);

            // op::tanh(hz, hv);

            // la::gpu::to_device(z, hz);

            op::gpu::tanh(z, v);
        }

        void tanh_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::tensor_like<double>>(t);
            auto& output = get_output<la::gpu::tensor_like<double>>(t);

            auto ch = get_child(t, 0);
            auto& ch_t = get_output<la::gpu::tensor_like<double>>(ch);

            if (ch->grad_needed && ch->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, ch_t);
                ch->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& result = get_grad<la::gpu::tensor_like<double>>(ch);

            if (ch->grad_needed) {
                // la::cpu::tensor<double> hresult = la::gpu::to_host(result);
                // la::cpu::tensor<double> hgrad = la::gpu::to_host(grad);
                // la::cpu::tensor<double> houtput = la::gpu::to_host(output);

                // op::itanh_grad(hresult, hgrad, houtput);

                // la::gpu::to_device(result, hresult);

                op::gpu::itanh_grad(result, grad, output);
            }
        }

        void relu_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));

            if (t->output == nullptr) {
                la::gpu::tensor<double> z;
                la::gpu::resize_as(z, v);
                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(z));
            }

            auto& z = get_output<la::gpu::tensor_like<double>>(t);

            op::gpu::relu(z, v);
        }

        void relu_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::tensor_like<double>>(t);
            auto& output = get_output<la::gpu::tensor_like<double>>(t);

            auto ch = get_child(t, 0);
            auto& ch_t = get_output<la::gpu::tensor_like<double>>(ch);

            if (ch->grad_needed && ch->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, ch_t);
                ch->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& result = get_grad<la::gpu::tensor_like<double>>(ch);

            if (ch->grad_needed) {
                op::gpu::irelu_grad(result, grad, output);
            }
        }

        void add_eval(std::shared_ptr<op_t> t)
        {
            auto& g = *t->graph;

            assert(g.adj[t->id].size() > 0);

            for (int i = 1; i < g.adj[t->id].size(); ++i) {
                if (get_output<la::gpu::tensor_like<double>>(get_child(t, i-1)).vec_size()
                        != get_output<la::gpu::tensor_like<double>>(get_child(t, i)).vec_size())
                {
                    std::cerr << get_output<la::gpu::tensor_like<double>>(get_child(t, i-1)).vec_size()
                        << " != " << get_output<la::gpu::tensor_like<double>>(
                            get_child(t, i)).vec_size() << std::endl;
                    exit(1);
                }
            }

            if (t->output == nullptr) {
                la::gpu::tensor<double> z;
                la::gpu::tensor_like<double>& m = get_output<la::gpu::tensor_like<double>>(
                    get_child(t, 0));
                la::gpu::resize_as(z, m);
                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(z));
            }

            auto& result = get_output<la::gpu::tensor_like<double>>(t);

            // la::cpu::tensor<double> hresult = la::gpu::to_host(result);

            for (int i = 0; i < g.adj[t->id].size(); ++i) {
                auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, i));

                // la::cpu::tensor<double> hu = la::gpu::to_host(u);

                // la::cpu::axpy(hresult, 1, hu);

                la::gpu::axpy(result, 1, u);
            }

            // la::gpu::to_device(result, hresult);
        }

        void add_grad(std::shared_ptr<op_t> t)
        {
            auto& g = *t->graph;

            auto& grad = get_grad<la::gpu::tensor_like<double>>(t);

            for (int i = 0; i < g.adj[t->id].size(); ++i) {
                auto c = get_child(t, i);

                if (c->grad_needed && c->grad == nullptr) {
                    auto& c_t = get_output<la::gpu::tensor_like<double>>(c);
                    la::gpu::tensor<double> g;
                    la::gpu::resize_as(g, c_t);
                    c->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
                }

                auto& u = get_grad<la::gpu::tensor_like<double>>(c);

                if (c->grad_needed) {
                    // la::cpu::tensor<double> hgrad = la::gpu::to_host(grad);
                    // la::cpu::tensor<double> hu = la::gpu::to_host(u);

                    // la::cpu::axpy(hu, 1, hgrad);

                    // la::gpu::to_device(u, hu);

                    la::gpu::axpy(u, 1, grad);
                }
            }
        }

        void add_to_eval(std::shared_ptr<op_t> t)
        {
            auto& g = *t->graph;

            assert(g.adj[t->id].size() > 0);

            for (int i = 2; i < g.adj[t->id].size(); ++i) {
                if (get_output<la::gpu::tensor_like<double>>(get_child(t, i-1)).vec_size()
                        != get_output<la::gpu::tensor_like<double>>(get_child(t, i)).vec_size())
                {
                    std::ostringstream oss;
                    oss << get_output<la::gpu::tensor_like<double>>(get_child(t, i-1)).vec_size()
                        << " != "
                        << get_output<la::gpu::tensor_like<double>>(get_child(t, i)).vec_size();
                    throw std::logic_error(oss.str());
                }
            }

            auto storage = get_child(t, 0);
            assert(storage->output != nullptr);

            auto& result = get_output<la::gpu::tensor_like<double>>(t);

            for (int i = 1; i < g.adj[t->id].size(); ++i) {
                auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, i));

                la::gpu::axpy(result, 1, u);
            }

            if (t->grad_needed && storage->grad == nullptr) {
                la::gpu::tensor<double> z;
                la::gpu::tensor_like<double>& m
                    = get_output<la::gpu::tensor_like<double>>(get_child(t, 1));
                la::gpu::resize_as(z, m);
                storage->grad = std::make_shared<la::gpu::tensor<double>>(std::move(z));
            }

            t->grad = storage->grad;
        }

        void add_to_grad(std::shared_ptr<op_t> t)
        {
            auto& g = *t->graph;

            auto& grad = get_grad<la::gpu::tensor_like<double>>(t);

            for (int i = 1; i < g.adj[t->id].size(); ++i) {
                auto c = get_child(t, i);

                if (c->grad_needed && c->grad == nullptr) {
                    auto& c_t = get_output<la::gpu::tensor_like<double>>(c);
                    la::gpu::tensor<double> g;
                    la::gpu::resize_as(g, c_t);
                    c->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
                }

                auto& u = get_grad<la::gpu::tensor_like<double>>(c);

                if (c->grad_needed) {
                    la::gpu::axpy(u, 1, grad);
                }
            }
        }
    
        void sub_eval(std::shared_ptr<op_t> t)
        {
            auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 1));

            if (t->output == nullptr) {
                la::gpu::tensor<double> z;
                la::gpu::resize_as(z, u);
                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(z));
            }

            auto& result = get_output<la::gpu::tensor_like<double>>(t);

            // la::cpu::tensor<double> hresult = la::gpu::to_host(result);
            // la::cpu::tensor<double> hu = la::gpu::to_host(u);
            // la::cpu::tensor<double> hv = la::gpu::to_host(v);

            // la::cpu::axpy(hresult, 1, hu);
            // la::cpu::axpy(hresult, -1, hv);

            // la::gpu::to_device(result, hresult);

            la::gpu::axpy(result, 1, u);
            la::gpu::axpy(result, -1, v);
        }

        void sub_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::tensor_like<double>>(t);

            auto u_o = get_child(t, 0);
            auto& u_t = get_output<la::gpu::tensor_like<double>>(u_o);

            if (u_o->grad_needed && u_o->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, u_t);
                u_o->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto v_o = get_child(t, 1);
            auto& v_t = get_output<la::gpu::tensor_like<double>>(v_o);

            if (v_o->grad_needed && v_o->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, v_t);
                v_o->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& u_grad = get_grad<la::gpu::tensor_like<double>>(u_o);
            auto& v_grad = get_grad<la::gpu::tensor_like<double>>(v_o);

            if (u_o->grad_needed) {
                // la::cpu::tensor<double> hu_grad = la::gpu::to_host(u_grad);
                // la::cpu::tensor<double> hgrad = la::gpu::to_host(grad);
                // la::cpu::axpy(hu_grad, 1, hgrad);

                // la::gpu::to_device(u_grad, hu_grad);

                la::gpu::axpy(u_grad, 1, grad);
            }

            if (v_o->grad_needed) {
                // la::cpu::tensor<double> hv_grad = la::gpu::to_host(v_grad);
                // la::cpu::tensor<double> hgrad = la::gpu::to_host(grad);
                // la::cpu::axpy(hv_grad, -1, hgrad);

                // la::gpu::to_device(v_grad, hv_grad);

                la::gpu::axpy(v_grad, -1, grad);
            }
        }

        void logsoftmax_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));

            if (t->output == nullptr) {
                la::gpu::tensor<double> z;
                la::gpu::resize_as(z, v);
                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(z));
            }

            auto& z = get_output<la::gpu::tensor_like<double>>(t);

            // la::cpu::tensor<double> hz = la::gpu::to_host(z);
            // la::cpu::tensor<double> hv = la::gpu::to_host(v);

            // op::logsoftmax(hz, hv);

            // la::gpu::to_device(z, hz);

            op::gpu::logsoftmax(z, v);
        }

        void logsoftmax_grad(std::shared_ptr<op_t> t)
        {
            auto& output = get_output<la::gpu::tensor_like<double>>(t);
            auto& grad = get_grad<la::gpu::tensor_like<double>>(t);

            auto ch = get_child(t, 0);
            auto& ch_t = get_output<la::gpu::tensor_like<double>>(ch);

            if (ch->grad_needed && ch->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, ch_t);
                ch->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& result = get_grad<la::gpu::tensor_like<double>>(ch);

            if (ch->grad_needed) {
                // la::cpu::tensor<double> hgrad = la::gpu::to_host(grad);
                // la::cpu::tensor<double> houtput = la::gpu::to_host(output);
                // la::cpu::tensor<double> hresult = la::gpu::to_host(result);

                // op::ilogsoftmax_grad(hresult, hgrad, houtput);

                // la::gpu::to_device(result, hresult);

                op::gpu::ilogsoftmax_grad(result, grad, output);
            }
        }

        void dot_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));
            auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, 1));

            // la::cpu::tensor<double> hv = la::gpu::to_host(v);
            // la::cpu::tensor<double> hu = la::gpu::to_host(u);
            // 
            // t->output = std::make_shared<double>(la::cpu::dot(hv, hu));

            t->output = std::make_shared<double>(la::gpu::dot(v, u));
        }

        void dot_grad(std::shared_ptr<op_t> t)
        {
            auto c0 = get_child(t, 0);
            auto c1 = get_child(t, 1);

            auto& v = get_output<la::gpu::tensor_like<double>>(c0);
            auto& u = get_output<la::gpu::tensor_like<double>>(c1);

            assert(v.vec_size() == u.vec_size());

            double grad = get_grad<double>(t);

            if (c0->grad_needed && c0->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, u);
                c0->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& v_grad = get_grad<la::gpu::tensor_like<double>>(c0);

            if (c0->grad_needed) {
                // la::cpu::tensor<double> hv_grad = la::gpu::to_host(v_grad);
                // la::cpu::tensor<double> hu = la::gpu::to_host(u);

                // la::cpu::axpy(hv_grad, grad, hu);

                // la::gpu::to_device(v_grad, hv_grad);

                la::gpu::axpy(v_grad, grad, u);
            }

            if (c1->grad_needed && c1->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, v);
                c1->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& u_grad = get_grad<la::gpu::tensor_like<double>>(c1);

            if (c1->grad_needed) {
                // la::cpu::tensor<double> hu_grad = la::gpu::to_host(u_grad);
                // la::cpu::tensor<double> hv = la::gpu::to_host(v);

                // la::cpu::axpy(hu_grad, grad, hv);

                // la::gpu::to_device(u_grad, hu_grad);

                la::gpu::axpy(u_grad, grad, v);
            }
        }

        void weak_cat_eval(std::shared_ptr<op_t> t)
        {
            auto& graph = *t->graph;

            auto storage = graph.vertices[graph.adj[t->id].back()];

            if (t->grad_needed && storage->grad == nullptr) {
                auto& storage_t = get_output<la::gpu::tensor_like<double>>(storage);
                la::gpu::tensor<double> zeros;
                la::gpu::resize_as(zeros, storage_t);
                storage->grad = std::make_shared<la::gpu::tensor<double>>(zeros);
                t->grad = storage->grad;

                unsigned long size = 0;
                for (int i = 0; i < graph.adj[t->id].size() - 1; ++i) {
                    auto ch = get_child(t, i);
                    auto& ch_t = get_output<la::gpu::tensor_like<double>>(ch);
                    la::gpu::weak_tensor<double> g { storage_t.data() + size, ch_t.sizes() };
                    ch->grad = std::make_shared<la::gpu::weak_tensor<double>>(g);
                    size += ch_t.vec_size();
                }

                assert(size == storage_t.vec_size());
            }

            t->grad = storage->grad;
        }

        void weak_cat_grad(std::shared_ptr<op_t> t)
        {
        }

        void row_cat_eval(std::shared_ptr<op_t> t)
        {
            auto& g = *t->graph;
            assert(g.adj[t->id].size() > 0);

            unsigned int rows = g.adj[t->id].size();

            auto& v0 = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));

            if (t->output == nullptr) {
                std::vector<unsigned int> sizes = v0.sizes();
                std::vector<unsigned int> new_sizes;
                new_sizes.push_back(rows);
                new_sizes.insert(new_sizes.end(), sizes.begin(), sizes.end());

                la::gpu::tensor<double> z;
                z.resize(new_sizes);
                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(z));
            }

            auto& z = get_output<la::gpu::tensor_like<double>>(t);

            la::gpu::weak_matrix<double> m { z.data(), rows, v0.vec_size() };

            for (int i = 0; i < m.rows(); ++i) {
                auto& vi = get_output<la::gpu::tensor_like<double>>(get_child(t, i));

                assert(vi.vec_size() == m.cols());

                la::gpu::weak_tensor<double> mi (z.data() + i * m.cols(), {m.cols()});

                la::gpu::copy(mi, vi);
            }
        }

        void row_cat_grad(std::shared_ptr<op_t> t)
        {
            auto& g = *t->graph;

            auto& z = autodiff::get_grad<la::gpu::tensor_like<double>>(t);

            assert(z.size(0) == g.adj[t->id].size());

            la::gpu::weak_matrix<double> m { z.data(), z.size(0), z.vec_size() / z.size(0) };

            for (int i = 0; i < m.rows(); ++i) {
                auto c = get_child(t, i);

                auto& v = autodiff::get_output<la::gpu::tensor_like<double>>(c);

                assert(v.vec_size() == m.cols());

                if (c->grad_needed && c->grad == nullptr) {
                    la::gpu::tensor<double> g;
                    la::gpu::resize_as(g, v);
                    c->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
                }

                auto& g = autodiff::get_grad<la::gpu::tensor_like<double>>(c);

                if (c->grad_needed) {
                    la::gpu::axpy(g.as_vector(), 1,
                        la::gpu::weak_vector<double>(z.data() + i * m.cols(), m.cols()));
                }
            }
        }

        void reshape_eval(std::shared_ptr<op_t> t)
        {
            std::vector<unsigned int>& sizes = *std::static_pointer_cast<std::vector<unsigned int>>(t->data);
            la::gpu::tensor_like<double>& input = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));

            unsigned int d = (sizes.size() == 0 ? 0 : 1);
            for (int i = 0; i < sizes.size(); ++i) {
                d *= sizes[i];
            }
            assert(d <= input.vec_size());

            la::gpu::weak_tensor<double> result { input.data(), sizes };
            t->output = std::make_shared<la::gpu::weak_tensor<double>>(result);

            auto c = get_child(t, 0);

            if (c->grad_needed && c->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, input);
                c->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            if (c->grad_needed) {
                auto& g = autodiff::get_grad<la::gpu::tensor_like<double>>(c);
                la::gpu::weak_tensor<double> wg { g.data(), sizes };
                t->grad = std::make_shared<la::gpu::weak_tensor<double>>(wg);
            }
        }

        void reshape_grad(std::shared_ptr<op_t> t)
        {
        }

        void resize_as_eval(std::shared_ptr<op_t> t)
        {
            auto c = get_child(t, 0);

            if (t->output == nullptr) {
                auto& c_t = get_output<la::gpu::tensor_like<double>>(c);

                double value = *std::static_pointer_cast<double>(t->data);

                la::gpu::tensor<double> w;
                la::gpu::resize_as(w, c_t, value);

                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(w));
            }
        }

        void resize_as_grad(std::shared_ptr<op_t> t)
        {
        }

        void rep_row_to_eval(std::shared_ptr<op_t> t)
        {
            auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 1));

            assert(v.vec_size() % u.vec_size() == 0);

            if (t->output == nullptr) {
                la::gpu::tensor<double> w;
                std::vector<unsigned int> sizes = u.sizes();
                std::vector<unsigned int> new_sizes;
                new_sizes.push_back(v.vec_size() / u.vec_size());
                new_sizes.insert(new_sizes.end(), sizes.begin(), sizes.end());
                w.resize(new_sizes);

                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(w));
            }

            auto& w = get_output<la::gpu::tensor_like<double>>(t);

            la::gpu::weak_matrix<double> w_mat {w.data(), v.vec_size() / u.vec_size(), u.vec_size()};

            la::gpu::vector<double> one;
            one.resize(v.vec_size() / u.vec_size(), 1);

            la::gpu::outer_prod(w_mat, one, u.as_vector());

            // la::cpu::tensor<double> hw = la::gpu::to_host(w);
            // la::cpu::tensor<double> hu = la::gpu::to_host(u);

            // la::cpu::weak_matrix<double> hw_mat {hw.data(), v.vec_size() / u.vec_size(), u.vec_size()};

            // la::cpu::vector<double> one;
            // one.resize(v.vec_size() / u.vec_size(), 1);

            // la::cpu::outer_prod(hw_mat, one, hu.as_vector());

            // la::gpu::to_device(w, hw);
        }

        void rep_row_to_grad(std::shared_ptr<op_t> t)
        {
            auto u_op = get_child(t, 0);

            auto& u = get_output<la::gpu::tensor_like<double>>(u_op);
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 1));

            if (u_op->grad_needed && u_op->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, u);
                u_op->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& g_w = get_grad<la::gpu::tensor_like<double>>(t);
            auto& g_u = get_grad<la::gpu::tensor_like<double>>(u_op);

            if (u_op->grad_needed) {
                // la::cpu::tensor<double> hg_w = la::gpu::to_host(g_w);
                // la::cpu::tensor<double> hg_u = la::gpu::to_host(g_u);

                // la::cpu::weak_matrix<double> hz {
                //     hg_w.data(), hg_w.vec_size() / u.vec_size(), u.vec_size()};

                // la::cpu::vector<double> one;
                // one.resize({hg_w.vec_size() / u.vec_size()}, 1);

                // la::cpu::lmul(hg_u.as_vector(), one, hz);

                // la::gpu::to_device(g_u, hg_u);

                la::gpu::weak_matrix<double> z {
                    g_w.data(), g_w.vec_size() / u.vec_size(), u.vec_size()};

                la::gpu::vector<double> one;
                one.resize({g_w.vec_size() / u.vec_size()}, 1);

                la::gpu::lmul(g_u.as_vector(), one, z);
            }
        }

        void rep_col_to_eval(std::shared_ptr<op_t> t)
        {
            auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 1));

            assert(v.vec_size() % u.vec_size() == 0);

            if (t->output == nullptr) {
                la::gpu::tensor<double> w;
                std::vector<unsigned int> sizes = u.sizes();
                sizes.push_back(v.vec_size() / u.vec_size());
                w.resize(sizes);
                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(w));
            }

            auto& w = get_output<la::gpu::tensor_like<double>>(t);

            la::gpu::vector<double> one;
            one.resize(v.vec_size() / u.vec_size(), 1);

            la::gpu::outer_prod(w.as_matrix(), u.as_vector(), one);
        }

        void rep_col_to_grad(std::shared_ptr<op_t> t)
        {
            auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 1));

            assert(v.vec_size() % u.vec_size() == 0);

            auto u_op = get_child(t, 0);

            if (u_op->grad_needed && u_op->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, u);
                u_op->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& g_w = get_grad<la::gpu::tensor_like<double>>(t);
            auto& g_u = get_grad<la::gpu::tensor_like<double>>(u_op);

            if (u_op->grad_needed) {
                la::gpu::vector<double> one;
                one.resize(v.vec_size() / u.vec_size(), 1);
                
                la::gpu::mul(g_u.as_vector(), g_w.as_matrix(), one);
            }
        }

        void dropout_mask_eval(std::shared_ptr<op_t> t)
        {
            auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));

            double prob;
            std::default_random_engine *gen;
            std::tie(prob, gen) = *std::static_pointer_cast<
                std::tuple<double, std::default_random_engine*>>(t->data);

            la::cpu::tensor<double> w;
            w.resize(u.sizes());

            std::bernoulli_distribution bernoulli { 1 - prob };

            double *w_data = w.data();

            for (int i = 0; i < w.vec_size(); ++i) {
                w_data[i] = bernoulli(*gen) / (1 - prob);
            }

            t->output = std::make_shared<la::gpu::tensor<double>>(la::gpu::tensor<double>(w));
        }

        void dropout_mask_grad(std::shared_ptr<op_t> t)
        {
        }

    }
}
