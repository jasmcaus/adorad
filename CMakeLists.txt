cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

project(ad CXX)

file(GLOB SOURCES "adorad/*.cpp")

# Adding the executable 
add_executable(ad main.cpp)

target_include_directories(ad
    PRIVATE
        ${PROJECT_SOURCE_DIR}/adorad/coreten
)
