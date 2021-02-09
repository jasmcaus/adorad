#include "autodiff/autodiff-op.h"
#include "ebt/ebt.h"
#include <cmath>
#include <cassert>
#include <limits>
#include <cblas.h>

namespace autodiff {

    namespace op {

        void logistic(la::cpu::tensor_like<double>& u, la::cpu::tensor_like<double> const& v)
        {
            assert(u.vec_size() == v.vec_size());

            double *u_data = u.data();
            double const *v_data = v.data();

            for (int i = 0; i < v.vec_size(); ++i) {
                u_data[i] = 1 / (1 + std::exp(-v_data[i]));
            }
        }

        void ilogistic_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& output)
        {
            assert(result.vec_size() == grad.vec_size() && grad.vec_size() == output.vec_size());

            double *result_data = result.data();
            double const *grad_data = grad.data();
            double const *output_data = output.data();

            for (int i = 0; i < output.vec_size(); ++i) {
                result_data[i] += grad_data[i] * output_data[i] * (1 - output_data[i]);
            }
        }

        void relu(la::cpu::tensor_like<double>& u, la::cpu::tensor_like<double> const& v)
        {
            assert(u.vec_size() == v.vec_size());

            double *u_data = u.data();
            double const *v_data = v.data();

            for (int i = 0; i < v.vec_size(); ++i) {
                u_data[i] = std::max(0.0, v_data[i]);
            }
        }

        void irelu_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& output)
        {
            assert(result.vec_size() == grad.vec_size() && grad.vec_size() == output.vec_size());

            double *result_data = result.data();
            double const *grad_data = grad.data();
            double const *output_data = output.data();

            for (int i = 0; i < output.vec_size(); ++i) {
                result_data[i] += (output_data[i] > 0 ? grad_data[i] : 0);
            }
        }

        void tanh(la::cpu::tensor_like<double>& u, la::cpu::tensor_like<double> const& v)
        {
            assert(u.vec_size() == v.vec_size());

            double *u_data = u.data();
            double const *v_data = v.data();

            for (int i = 0; i < v.vec_size(); ++i) {
                if (v_data[i] > 0) {
                    double z = std::exp(-2 * v_data[i]);
                    u_data[i] = (1 - z) / (1 + z);
                } else {
                    double z = std::exp(2 * v_data[i]);
                    u_data[i] = (z - 1) / (z + 1);
                }
            }
        }

        void itanh_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& output)
        {
            assert(result.vec_size() == grad.vec_size() && grad.vec_size() == output.vec_size());

            double *result_data = result.data();
            double const *grad_data = grad.data();
            double const *output_data = output.data();

            for (int i = 0; i < output.vec_size(); ++i) {
                result_data[i] += grad_data[i] * (1 - output_data[i] * output_data[i]);
            }
        }

        void softmax(la::cpu::tensor_like<double>& u, la::cpu::tensor_like<double> const& v)
        {
            assert(u.vec_size() == v.vec_size());

            la::cpu::matrix_like<double> const& m = v.as_matrix();

            la::cpu::vector<double> logZs;
            logZs.resize(m.rows(), -std::numeric_limits<double>::infinity());

            for (int i = 0; i < m.rows(); ++i) {
                for (int j = 0; j < m.cols(); ++j) {
                    logZs(i) = ebt::log_add(logZs(i), m(i, j));
                }
            }

            la::cpu::weak_matrix<double> result {u.data(), m.rows(), m.cols()};

            for (int i = 0; i < result.rows(); ++i) {
                for (int j = 0; j < result.cols(); ++j) {
                    result(i, j) = std::exp(m(i, j) - logZs(i));
                }
            }
        }

        void isoftmax_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& output)
        {
            assert(result.vec_size() == grad.vec_size() && grad.vec_size() == output.vec_size());

            la::cpu::tensor<double> mu;
            std::vector<unsigned int> sizes = grad.sizes();
            sizes.pop_back();
            mu.resize(sizes);

            la::cpu::vdot(mu, grad, output);

            la::cpu::matrix_like<double> const& output_m = output.as_matrix();

            double *result_data = result.data();
            double const *output_data = output.data();
            double const *grad_data = grad.data();
            double const *mu_data = mu.data();

            for (int i = 0; i < output_m.rows(); ++i) {
                int c = i * output_m.cols();

                for (int j = 0; j < output_m.cols(); ++j) {
                    result_data[c + j] += output_data[c + j] * (grad_data[c + j] - mu_data[i]);
                }
            }
        }

        void spatial_softmax(la::cpu::tensor_like<double>& u, la::cpu::tensor_like<double> const& v)
        {
            assert(u.vec_size() == v.vec_size());
            assert(u.dim() == v.dim() && u.dim() == 4);

            double *u_data = u.data();
            double const *v_data = v.data();

            int batch_size = v.size(0);
            int time = v.size(1);
            int freq = v.size(2);
            int nchannel = v.size(3);

            la::cpu::matrix<double> logZ;
            logZ.resize(batch_size, nchannel, -std::numeric_limits<double>::infinity());

            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < time * freq; ++i) {
                    for (int j = 0; j < nchannel; ++j) {
                        logZ(b, j) = ebt::log_add(logZ(b, j),
                            v_data[b * time * freq * nchannel + i * nchannel + j]);
                    }
                }
            }

            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < time * freq; ++i) {
                    for (int j = 0; j < nchannel; ++j) {
                        u_data[b * time * freq * nchannel + i * nchannel + j] =
                            std::exp(v_data[b * time * freq * nchannel + i * nchannel + j] - logZ(b, j));
                    }
                }
            }
        }

        void ispatial_softmax_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& output)
        {
            assert(result.vec_size() == grad.vec_size() && grad.vec_size() == output.vec_size());
            assert(result.dim() == output.dim() && output.dim() == grad.dim() && output.dim() == 4);

            int batch_size = output.size(0);
            int time = output.size(1);
            int freq = output.size(2);
            int nchannel = output.size(3);

            la::cpu::matrix<double> mu;
            mu.resize(batch_size, nchannel);

            double *result_data = result.data();
            double const *output_data = output.data();
            double const *grad_data = grad.data();

            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < time * freq; ++i) {
                    for (int j = 0; j < nchannel; ++j) {
                        mu(b, j) +=
                            grad_data[b * time * freq * nchannel + i * nchannel + j];
                    }
                }
            }

            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < time * freq; ++i) {
                    for (int j = 0; j < nchannel; ++j) {
                        result_data[b * time * freq * nchannel + i * nchannel + j] +=
                            output_data[b * time * freq * nchannel + i * nchannel + j]
                            * (grad_data[b * time * freq * nchannel + i * nchannel + j] - mu(b, j));
                    }
                }
            }
        }

        void logsoftmax(la::cpu::tensor_like<double>& u, la::cpu::tensor_like<double> const& v)
        {
            assert(u.vec_size() == v.vec_size());

            la::cpu::matrix_like<double> const& m = v.as_matrix();

            la::cpu::vector<double> logZs;
            logZs.resize(m.rows(), -std::numeric_limits<double>::infinity());

            for (int i = 0; i < m.rows(); ++i) {
                for (int j = 0; j < m.cols(); ++j) {
                    logZs(i) = ebt::log_add(logZs(i), m(i, j));
                }
            }

            la::cpu::weak_matrix<double> result {u.data(), m.rows(), m.cols()};

            for (int i = 0; i < result.rows(); ++i) {
                for (int j = 0; j < result.cols(); ++j) {
                    result(i, j) = m(i, j) - logZs(i);
                }
            }
        }

        void ilogsoftmax_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& output)
        {
            assert(result.vec_size() == grad.vec_size() && grad.vec_size() == output.vec_size());

            la::cpu::matrix_like<double> const& grad_m = grad.as_matrix();

            la::cpu::weak_matrix<double> result_m {result.data(),
                grad_m.rows(), grad_m.cols()};

            la::cpu::weak_matrix<double> output_m {const_cast<double*>(output.data()),
                grad_m.rows(), grad_m.cols()};

            la::cpu::vector<double> mu;
            mu.resize(grad_m.rows(), 0);

            la::cpu::vector<double> one;
            one.resize(grad_m.cols(), 1);

            la::cpu::mul(mu, grad_m, one);

            for (int i = 0; i < grad_m.rows(); ++i) {
                for (int j = 0; j < grad_m.cols(); ++j) {
                    result_m(i, j) += grad_m(i, j) - std::exp(output_m(i, j)) * mu(i);
                }
            }
        }

        void pool_max(la::cpu::tensor_like<double>& indices,
            la::cpu::tensor_like<double>& output,
            la::cpu::tensor_like<double> const& input,
            int dim1, int dim2, int stride1, int stride2)
        {
            assert(input.dim() == 3);
            assert(output.dim() == 3);

            int d1 = input.size(0);
            int d2 = input.size(1);
            int channels = input.size(2);

            int f1 = output.size(0);
            int f2 = output.size(1);

            assert(channels == output.size(2));

            double const* input_data = input.data();
            double* output_data = output.data();
            double* indices_data = indices.data();

            int input_vec_size = input.vec_size();
            int output_vec_size = output.vec_size();
            int indices_vec_size = indices.vec_size();

            for (int i = 0; i < d1; i += stride1) {
                for (int j = 0; j < d2; j += stride2) {
                    for (int m = 0; m < dim1; ++m) {
                        for (int n = 0; n < dim2; ++n) {

                            int c1 = i + (m - dim1 / 2);
                            int c2 = j + (n - dim2 / 2);

                            if (c1 < 0 || c1 >= d1 || c2 < 0 || c2 >= d2) {
                                continue;
                            }

                            int input_base = c1 * d2 * channels + c2 * channels;

                            int u = i / stride1;
                            int v = j / stride2;

                            int output_base = u * f2 * channels + v * channels;

                            for (int c = 0; c < channels; ++c) {

                                assert(0 <= input_base + c && input_base + c < input_vec_size);
                                assert(0 <= output_base + c && output_base + c < output_vec_size);
                                assert(output_base + c < indices_vec_size);

                                double v = input_data[input_base + c];
                                double& max = output_data[output_base + c];

                                if (v > max) {
                                    max = v;
                                    indices_data[output_base + c] = m * dim2 + n;
                                }
                            }
                        }
                    }
                }
            }
        }

        void pool_max_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& indices,
            int dim1, int dim2, int stride1, int stride2)
        {
            assert(grad.dim() == 3);
            assert(result.dim() == 3);

            int f1 = grad.size(0);
            int f2 = grad.size(1);
            int channels = grad.size(2);

            int d1 = result.size(0);
            int d2 = result.size(1);

            assert(result.size(2) == channels);

            double* result_data = result.data();
            double const* grad_data = grad.data();
            double const* indices_data = indices.data();

            int grad_vec_size = grad.vec_size();
            int result_vec_size = result.vec_size();

            for (int u = 0; u < f1; ++u) {
                for (int v = 0; v < f2; ++v) {

                    int grad_base = u * f2 * channels + v * channels;

                    for (int c = 0; c < channels; ++c) {

                        int m = int(indices_data[grad_base + c]) / dim2;
                        int n = int(indices_data[grad_base + c]) % dim2;

                        assert(0 <= m && m < dim1);
                        assert(0 <= n && n < dim2);

                        int i = u * stride1;
                        int j = v * stride2;

                        assert(0 <= i && i <= d1);
                        assert(0 <= j && j <= d2);

                        assert(0 <= (i + m - dim1 / 2) && (i + m - dim1 / 2) < d1);
                        assert(0 <= (j + n - dim2 / 2) && (j + n - dim2 / 2) < d2);

                        int result_base = (i + m - dim1 / 2) * d2 * channels + (j + n - dim2 / 2) * channels;

                        assert(0 <= grad_base + c && grad_base + c < grad_vec_size);
                        assert(0 <= result_base + c && result_base + c < result_vec_size);

                        result_data[result_base + c] += grad_data[grad_base + c];
                    }

                }
            }
        }

    }
}
