cmake_minimum_required(VERSION 3.7)

# project name
project(adorad)

add_library(
    adorad 
    adorad/src/Tensor.h
    adorad/src/TensorFuncs.h
    adorad/src/m.cpp
)

# add_excutable(adorad adorad/main.cpp)