
set(PYTHON_LIB pytensor)

add_library(${PYTHON_LIB} MODULE
        src/tensor.cpp)

pybind11_extension(${PYTHON_LIB})

pybind11_strip(${PYTHON_LIB})

set_target_properties(${PYTHON_LIB} PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden")

target_link_libraries(${PYTHON_LIB} PRIVATE
        pybind11::module
        pybind11::lto
        tensor
        )
