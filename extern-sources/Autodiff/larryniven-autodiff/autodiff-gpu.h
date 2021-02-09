#ifndef AUTODIFF_GPU_H
#define AUTODIFF_GPU_H

#include "autodiff/autodiff.h"
#include "la/la-gpu.h"

namespace autodiff {

    namespace gpu {

        void zeros_eval(std::shared_ptr<op_t> t);
        void zeros_grad(std::shared_ptr<op_t> t);

        void weak_var_eval(std::shared_ptr<op_t> t);
        void weak_var_grad(std::shared_ptr<op_t> t);

        void subtensor_eval(std::shared_ptr<op_t> t);
        void subtensor_grad(std::shared_ptr<op_t> t);

        void split_block_eval(std::shared_ptr<op_t> t);
        void split_block_grad(std::shared_ptr<op_t> t);

        void mul_eval(std::shared_ptr<op_t> t);
        void mul_grad(std::shared_ptr<op_t> t);

        void emul_eval(std::shared_ptr<op_t> t);
        void emul_grad(std::shared_ptr<op_t> t);

        void emul_to_eval(std::shared_ptr<op_t> t);
        void emul_to_grad(std::shared_ptr<op_t> t);

        void logistic_eval(std::shared_ptr<op_t> t);
        void logistic_grad(std::shared_ptr<op_t> t);

        void tanh_eval(std::shared_ptr<op_t> t);
        void tanh_grad(std::shared_ptr<op_t> t);

        void relu_eval(std::shared_ptr<op_t> t);
        void relu_grad(std::shared_ptr<op_t> t);

        void add_eval(std::shared_ptr<op_t> t);
        void add_grad(std::shared_ptr<op_t> t);

        void add_to_eval(std::shared_ptr<op_t> t);
        void add_to_grad(std::shared_ptr<op_t> t);

        void sub_eval(std::shared_ptr<op_t> t);
        void sub_grad(std::shared_ptr<op_t> t);

        void logsoftmax_eval(std::shared_ptr<op_t> t);
        void logsoftmax_grad(std::shared_ptr<op_t> t);

        void dot_eval(std::shared_ptr<op_t> t);
        void dot_grad(std::shared_ptr<op_t> t);

        void weak_cat_eval(std::shared_ptr<op_t> t);
        void weak_cat_grad(std::shared_ptr<op_t> t);

        void row_cat_eval(std::shared_ptr<op_t> t);
        void row_cat_grad(std::shared_ptr<op_t> t);

        void resize_as_eval(std::shared_ptr<op_t> t);
        void resize_as_grad(std::shared_ptr<op_t> t);

        void rep_row_to_eval(std::shared_ptr<op_t> t);
        void rep_row_to_grad(std::shared_ptr<op_t> t);

        void rep_col_to_eval(std::shared_ptr<op_t> t);
        void rep_col_to_grad(std::shared_ptr<op_t> t);

        void dropout_mask_eval(std::shared_ptr<op_t> t);
        void dropout_mask_grad(std::shared_ptr<op_t> t);

        static std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> eval_funcs {
            { "zeros", zeros_eval },
            { "var", var_eval },
            { "weak_var", weak_var_eval },
            { "subtensor", subtensor_eval },
            { "split_block", split_block_eval },
            { "mul", mul_eval },
            { "emul", emul_eval },
            { "emul_to", emul_to_eval },
            { "logistic", logistic_eval },
            { "tanh", tanh_eval },
            { "relu", relu_eval },
            { "add", add_eval },
            { "add_to", add_to_eval },
            { "sub", sub_eval },
            { "logsoftmax", logsoftmax_eval },
            { "dot", dot_eval },
            { "weak_cat", weak_cat_eval },
            { "row_cat", row_cat_eval },
            { "resize_as", resize_as_eval },
            { "rep_row_to", rep_row_to_eval },
            { "rep_col_to", rep_col_to_eval },
            { "dropout_mask", dropout_mask_eval },
        };

        static std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> grad_funcs {
            { "zeros", zeros_grad },
            { "var", var_grad },
            { "weak_var", weak_var_grad },
            { "subtensor", subtensor_grad },
            { "split_block", split_block_grad },
            { "mul", mul_grad },
            { "emul", emul_grad },
            { "emul_to", emul_to_grad },
            { "logistic", logistic_grad },
            { "tanh", tanh_grad },
            { "relu", relu_grad },
            { "add", add_grad },
            { "add_to", add_to_grad },
            { "sub", sub_grad },
            { "logsoftmax", logsoftmax_grad },
            { "dot", dot_grad },
            { "weak_cat", weak_cat_grad },
            { "row_cat", row_cat_grad },
            { "resize_as", resize_as_grad },
            { "rep_row_to", rep_row_to_grad },
            { "rep_col_to", rep_col_to_grad },
            { "dropout_mask", dropout_mask_grad },
        };

    }
}

#endif
