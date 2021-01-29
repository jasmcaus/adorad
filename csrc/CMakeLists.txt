#
# This is a CMake makefile.  CMake is a tool that helps you build C++ programs.
# You can download CMake from http://www.cmake.org.  This CMakeLists.txt file
# builds adorad's internal testing program.
#
cmake_minimum_required(VERSION 3.7)

# Every project needs a name.
project(ad)

file(GLOB SOURCES "adorad/*.cpp")

# Adding the executable 
add_executable(ad main.cpp)

target_include_directories(ad
    PRIVATE
        ${PROJECT_SOURCE_DIR}/adorad/coreten
)
