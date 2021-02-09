#ifndef AUTODIFF_OP_GPU_H
#define AUTODIFF_OP_GPU_H

#include "la/la-gpu.h"

namespace autodiff {

    namespace op {

        namespace gpu {

            void iouter_prod(la::gpu::matrix_like<double>& result,
                la::gpu::vector_like<double> const& x,
                la::gpu::vector_like<double> const& y);

            void ilmul(la::gpu::vector_like<double>& result,
                la::gpu::matrix_like<double> const& a,
                la::gpu::vector_like<double> const& x);

            void logistic(la::gpu::tensor_like<double>& u,
                la::gpu::tensor_like<double> const& v);

            void ilogistic_grad(la::gpu::tensor_like<double>& result,
                la::gpu::tensor_like<double> const& grad,
                la::gpu::tensor_like<double> const& output);

            void relu(la::gpu::tensor_like<double>& u,
                la::gpu::tensor_like<double> const& v);

            void irelu_grad(la::gpu::tensor_like<double>& result,
                la::gpu::tensor_like<double> const& grad,
                la::gpu::tensor_like<double> const& output);

            void tanh(la::gpu::tensor_like<double>& u,
                la::gpu::tensor_like<double> const& v);

            void itanh_grad(la::gpu::tensor_like<double>& result,
                la::gpu::tensor_like<double> const& grad,
                la::gpu::tensor_like<double> const& output);

            void softmax(la::gpu::vector_like<double>& u,
                la::gpu::vector_like<double> const& v);

            void isoftmax_grad(la::gpu::vector_like<double>& result,
                la::gpu::vector_like<double> const& grad,
                la::gpu::vector_like<double> const& output);

            void logsoftmax(la::gpu::tensor_like<double>& u,
                la::gpu::tensor_like<double> const& v);

            void ilogsoftmax_grad(la::gpu::tensor_like<double>& result,
                la::gpu::tensor_like<double> const& grad,
                la::gpu::tensor_like<double> const& output);
        }
    }
}

#endif
