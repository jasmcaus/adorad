#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tensor/tensor.hpp>

namespace py = pybind11;

using size_type = ts::Tensor<float, 2>::size_type;

PYBIND11_MODULE(pytensor, m)
{
    py::class_<ts::Tensor<float, 2>>(m, "Tensor2F", py::buffer_protocol())
        .def(py::init<size_type , size_type>())
        /// Construct from a buffer
        .def(py::init([](py::buffer const b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<float>::format() || info.ndim != 2)
                throw std::runtime_error("Incompatible buffer format!");

            auto v = new ts::Tensor<float, 2>(info.shape[0], info.shape[1]);
            memcpy(v->data()->data(),
                   info.ptr,
                   sizeof(float) * (size_t)(v->shape(0) * v->shape(1)));
            return v;
        }))
        .def("shape", [](ts::Tensor<float, 2> const  &t) -> std::array<int, 2> {
          return t.shape();
        })
        .def("data_size", &ts::Tensor<float, 2>::data_size)
        /// Bare bones interface
        .def("__getitem__",
             [](ts::Tensor<float, 2> const &t, std::pair<py::ssize_t, py::ssize_t> i) {
                 if (i.first >= t.shape(0) || i.second >= t.shape(1))
                     throw py::index_error();
                 return t(i.first, i.second);
             })
        .def("__setitem__",
             [](ts::Tensor<float, 2> &t, std::pair<py::ssize_t, py::ssize_t> i, float v) {
                 if (i.first >= t.shape(0) || i.second >= t.shape(1))
                     throw py::index_error();
                 t(i.first, i.second) = v;
             })
        /// Provide buffer access
        .def_buffer([](ts::Tensor<float, 2> &t) -> py::buffer_info {
            return py::buffer_info(
                t.data()->data(),                          /* Pointer to buffer */
                {t.shape(0), t.shape(1)},              /* Buffer dimensions */
                {sizeof(float) * size_t(t.shape(1)), /* Strides (in bytes) for each index */
                 sizeof(float)});
        });
}
