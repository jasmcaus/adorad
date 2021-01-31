#ifndef AUTODIFF_H
#define AUTODIFF_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <random>
#include <functional>

namespace autodiff {

    struct op_t;
    struct computation_graph;

    struct op_t {
        op_t();

        int id;
        bool grad_needed;

        std::shared_ptr<void> output;
        std::shared_ptr<void> grad;
        std::shared_ptr<void> data;

        std::string name;

        computation_graph *graph;
    };

    struct computation_graph {
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> eval_funcs;
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> grad_funcs;

        bool lazy;

        std::vector<std::shared_ptr<op_t>> vertices; 
        std::vector<std::vector<int>> adj;

        computation_graph();
        computation_graph(computation_graph const& graph);

        computation_graph& operator=(computation_graph const& other);

        template <class T>
        std::shared_ptr<op_t> var(T&& t)
        {
            std::shared_ptr<op_t> result = make_node("var");

            result->output = std::make_shared<typename std::decay<T>::type>(std::forward<T>(t));

            return result;
        }

        std::shared_ptr<op_t> var();

        std::shared_ptr<op_t> make_node(std::string name);

        void add_edge(std::shared_ptr<op_t> const& tail, std::shared_ptr<op_t> const& head);
    };

    std::shared_ptr<op_t> get_child(std::shared_ptr<op_t> const& t, int index);
    void add_child(std::shared_ptr<op_t> const& t, std::shared_ptr<op_t> const& ch);

    inline void var_eval(std::shared_ptr<op_t> t) {};
    inline void var_grad(std::shared_ptr<op_t> t) {};

    std::shared_ptr<op_t> zeros(computation_graph& graph, std::vector<unsigned int> sizes);
    void zeros_eval(std::shared_ptr<op_t> t);
    void zeros_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> weak_var(std::shared_ptr<op_t> t,
        int shift, std::vector<unsigned int> sizes);
    void weak_var_eval(std::shared_ptr<op_t> t);
    void weak_var_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> subtensor(std::shared_ptr<op_t> t,
        std::vector<unsigned int> shift,
        std::vector<unsigned int> sizes);
    void subtensor_eval(std::shared_ptr<op_t> t);
    void subtensor_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> split_block(std::shared_ptr<op_t> t, int block);
    void split_block_eval(std::shared_ptr<op_t> t);
    void split_block_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> sum(std::shared_ptr<op_t> t);
    void sum_eval(std::shared_ptr<op_t> t);
    void sum_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> mul(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2);
    void mul_eval(std::shared_ptr<op_t> t);
    void mul_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> ltmul(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2);
    void ltmul_eval(std::shared_ptr<op_t> t);
    void ltmul_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> rtmul(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2);
    void rtmul_eval(std::shared_ptr<op_t> t);
    void rtmul_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> emul(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2);
    void emul_eval(std::shared_ptr<op_t> t);
    void emul_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> emul_to(std::shared_ptr<op_t> s,
        std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2);
    void emul_to_eval(std::shared_ptr<op_t> t);
    void emul_to_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> ediv(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2);
    void ediv_eval(std::shared_ptr<op_t> t);
    void ediv_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> logistic(std::shared_ptr<op_t> input);
    void logistic_eval(std::shared_ptr<op_t> t);
    void logistic_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> relu(std::shared_ptr<op_t> input);
    void relu_eval(std::shared_ptr<op_t> t);
    void relu_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> tanh(std::shared_ptr<op_t> input);
    void tanh_eval(std::shared_ptr<op_t> t);
    void tanh_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> exp(std::shared_ptr<op_t> input);
    void exp_eval(std::shared_ptr<op_t> t);
    void exp_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> log(std::shared_ptr<op_t> input);
    void log_eval(std::shared_ptr<op_t> t);
    void log_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> add_to(std::shared_ptr<op_t> t, std::vector<std::shared_ptr<op_t>> ts);
    void add_to_eval(std::shared_ptr<op_t> t);
    void add_to_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> add(std::vector<std::shared_ptr<op_t>> t);
    std::shared_ptr<op_t> add(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2);
    void add_eval(std::shared_ptr<op_t> t);
    void add_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> sub(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2);
    void sub_eval(std::shared_ptr<op_t> t);
    void sub_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> norm(std::shared_ptr<op_t> t);
    void norm_eval(std::shared_ptr<op_t> t);
    void norm_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> softmax(std::shared_ptr<op_t> t);
    void softmax_eval(std::shared_ptr<op_t> t);
    void softmax_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> spatial_softmax(std::shared_ptr<op_t> t);
    void spatial_softmax_eval(std::shared_ptr<op_t> t);
    void spatial_softmax_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> logsoftmax(std::shared_ptr<op_t> t);
    void logsoftmax_eval(std::shared_ptr<op_t> t);
    void logsoftmax_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> dot(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2);
    void dot_eval(std::shared_ptr<op_t> t);
    void dot_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> weak_cat(std::vector<std::shared_ptr<op_t>> const& ts,
        std::shared_ptr<op_t> storage);
    void weak_cat_eval(std::shared_ptr<op_t> t);
    void weak_cat_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> row_cat(std::vector<std::shared_ptr<op_t>> const& row_vecs);
    void row_cat_eval(std::shared_ptr<op_t> t);
    void row_cat_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> col_cat(std::vector<std::shared_ptr<op_t>> const& col_vecs);
    void col_cat_eval(std::shared_ptr<op_t> t);
    void col_cat_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> row_at(std::shared_ptr<op_t> const& t, int i);
    void row_at_eval(std::shared_ptr<op_t> t);
    void row_at_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> reshape(std::shared_ptr<op_t> const& t, std::vector<unsigned int> sizes);
    void reshape_eval(std::shared_ptr<op_t> t);
    void reshape_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> resize_as(std::shared_ptr<op_t> const& t, double value = 0);
    void resize_as_eval(std::shared_ptr<op_t> t);
    void resize_as_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> rep_row_to(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2);
    void rep_row_to_eval(std::shared_ptr<op_t> t);
    void rep_row_to_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> rep_col_to(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2);
    void rep_col_to_eval(std::shared_ptr<op_t> t);
    void rep_col_to_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> corr_linearize(std::shared_ptr<op_t> const& t1, std::shared_ptr<op_t> t2,
        int p1, int p2, int d1, int d2);
    std::shared_ptr<op_t> corr_linearize(std::shared_ptr<op_t> const& t1, std::shared_ptr<op_t> t2);
    void corr_linearize_eval(std::shared_ptr<op_t> t);
    void corr_linearize_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> corr(std::shared_ptr<op_t> const& t1, std::shared_ptr<op_t> t2);

    std::shared_ptr<op_t> corr_delinearize(std::shared_ptr<op_t> const& t1, std::shared_ptr<op_t> t2,
        int p1, int p2, int d1, int d2);
    std::shared_ptr<op_t> corr_delinearize(std::shared_ptr<op_t> const& t1, std::shared_ptr<op_t> t2);
    void corr_delinearize_eval(std::shared_ptr<op_t> t);
    void corr_delinearize_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> pool_max(std::shared_ptr<op_t> t, int d1, int d2, int s1, int s2);
    void pool_max_eval(std::shared_ptr<op_t> t);
    void pool_max_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> seg_max(std::shared_ptr<op_t> t);
    void seg_max_eval(std::shared_ptr<op_t> t);
    void seg_max_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> high_pass_k(std::shared_ptr<op_t> t, int k);
    void high_pass_k_eval(std::shared_ptr<op_t> t);
    void high_pass_k_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> uniform(std::shared_ptr<op_t> t,
        std::default_random_engine& gen);
    void uniform_eval(std::shared_ptr<op_t> t);
    void uniform_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> normal(std::shared_ptr<op_t> t,
        std::default_random_engine& gen);
    void normal_eval(std::shared_ptr<op_t> t);
    void normal_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> dropout_mask(std::shared_ptr<op_t> t, double prob,
        std::default_random_engine& gen);
    void dropout_mask_eval(std::shared_ptr<op_t> t);
    void dropout_mask_grad(std::shared_ptr<op_t> t);

    std::vector<std::shared_ptr<op_t>> natural_topo_order(computation_graph const& graph);

    std::vector<std::shared_ptr<op_t>> topo_order(std::vector<std::shared_ptr<op_t>> const& roots,
        std::vector<std::shared_ptr<op_t>> const& boundaries);
    std::vector<std::shared_ptr<op_t>> topo_order(std::vector<std::shared_ptr<op_t>> const& roots);
    std::vector<std::shared_ptr<op_t>> topo_order(std::shared_ptr<op_t> const& root);

    void eval_vertex(std::shared_ptr<op_t> const& t,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs);

    void eval(std::vector<std::shared_ptr<op_t>> const& topo_order,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs);

    void eval(std::shared_ptr<op_t> const& root,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs);

    void grad(std::vector<std::shared_ptr<op_t>> const& topo_order,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs);

    void grad(std::shared_ptr<op_t> const& root,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs);

    void guarded_grad(std::vector<std::shared_ptr<op_t>> const& topo_order,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs);

    void guarded_grad(std::shared_ptr<op_t> const& root,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs);

    void clear_grad(std::vector<std::shared_ptr<op_t>> const& topo_order);
    void clear_grad(std::shared_ptr<op_t> const& root);

    template <class T>
    T& get_output(std::shared_ptr<op_t> t)
    {
        return *static_cast<T*>(t->output.get());
    }

    template <class T>
    T& get_grad(std::shared_ptr<op_t> t)
    {
        return *static_cast<T*>(t->grad.get());
    }

    static std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> eval_funcs {
        { "zeros", zeros_eval },
        { "var", var_eval },
        { "weak_var", weak_var_eval },
        { "subtensor", subtensor_eval },
        { "split_block", split_block_eval },
        { "sum", sum_eval },
        { "mul", mul_eval },
        { "ltmul", ltmul_eval },
        { "rtmul", rtmul_eval },
        { "emul", emul_eval },
        { "emul_to", emul_to_eval },
        { "ediv", ediv_eval },
        { "logistic", logistic_eval },
        { "relu", relu_eval },
        { "tanh", tanh_eval },
        { "exp", exp_eval },
        { "log", log_eval },
        { "add", add_eval },
        { "add_to", add_to_eval },
        { "sub", sub_eval },
        { "norm", norm_eval },
        { "softmax", softmax_eval },
        { "spatial_softmax", spatial_softmax_eval },
        { "logsoftmax", logsoftmax_eval },
        { "dot", dot_eval },
        { "weak_cat", weak_cat_eval },
        { "row_cat", row_cat_eval },
        { "col_cat", col_cat_eval },
        { "row_at", row_at_eval },
        { "reshape", reshape_eval },
        { "resize_as", resize_as_eval },
        { "rep_row_to", rep_row_to_eval },
        { "rep_col_to", rep_col_to_eval },
        { "corr_linearize", corr_linearize_eval },
        { "corr_delinearize", corr_delinearize_eval },
        { "pool_max", pool_max_eval },
        { "seg-max", seg_max_eval },
        { "high-pass-k", high_pass_k_eval },
        { "uniform", uniform_eval },
        { "normal", normal_eval },
        { "dropout_mask", dropout_mask_eval },
    };

    static std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> grad_funcs {
        { "zeros", zeros_grad },
        { "var", var_grad },
        { "weak_var", weak_var_grad },
        { "subtensor", subtensor_grad },
        { "split_block", split_block_grad },
        { "sum", sum_grad },
        { "mul", mul_grad },
        { "ltmul", ltmul_grad },
        { "rtmul", rtmul_grad },
        { "emul", emul_grad },
        { "emul_to", emul_to_grad },
        { "ediv", ediv_grad },
        { "logistic", logistic_grad },
        { "relu", relu_grad },
        { "tanh", tanh_grad },
        { "exp", exp_grad },
        { "log", log_grad },
        { "add", add_grad },
        { "add_to", add_to_grad },
        { "sub", sub_grad },
        { "norm", norm_grad },
        { "softmax", softmax_grad },
        { "spatial_softmax", spatial_softmax_grad },
        { "logsoftmax", logsoftmax_grad },
        { "dot", dot_grad },
        { "weak_cat", weak_cat_eval },
        { "row_cat", row_cat_grad },
        { "col_cat", col_cat_grad },
        { "row_at", row_at_grad },
        { "reshape", reshape_grad },
        { "resize_as", resize_as_grad },
        { "rep_row_to", rep_row_to_grad },
        { "rep_col_to", rep_col_to_grad },
        { "corr_linearize", corr_linearize_grad },
        { "corr_delinearize", corr_delinearize_grad },
        { "pool_max", pool_max_grad },
        { "seg-max", seg_max_grad },
        { "high-pass-k", high_pass_k_grad },
        { "uniform", uniform_grad },
        { "normal", normal_grad },
        { "dropout_mask", dropout_mask_grad },
    };

}

#endif
