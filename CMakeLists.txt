cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

project(adorad CXX)

set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 11)


# Disable in-source builds to prevent corruption of the source tree 
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "FATAL: In-source builds are not allowed. 
            You should create a separate directory for build files ")
endif()

# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)
if(env_cxx_standard GREATER -1)
    message(
        WARNING "C++ standard version definition detected in environment variable."
        "Adorad requires -std=c++17. Please remove -std=c++ settings in your environment.")
endif()

if(MSVC)
    set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
endif() 

# Break in case of popular CMake configuration mistakes
if(NOT CMAKE_SIZEOF_VOID_P GREATER 0)
    message(FATAL_ERROR "CMake failed to determine the bitness of the target platform. Check your CMake and compiler        
            installation. If you are cross-compiling, ensure that your CMake toolchain file correctly sets the 
            compiler details")
endif()

# Add these standard paths to the search paths for FIND_LIBRARY
# to find libraries from these locations first
if(UNIX AND NOT ANDROID)
  if(X86_64 OR CMAKE_SIZEOF_VOID_P EQUAL 8)
    if(EXISTS /lib64)
      list(APPEND CMAKE_LIBRARY_PATH /lib64)
    else()
      list(APPEND CMAKE_LIBRARY_PATH /lib)
    endif()
    if(EXISTS /usr/lib64)
      list(APPEND CMAKE_LIBRARY_PATH /usr/lib64)
    else()
      list(APPEND CMAKE_LIBRARY_PATH /usr/lib)
    endif()
  elseif(X86 OR CMAKE_SIZEOF_VOID_P EQUAL 4)
    if(EXISTS /lib32)
      list(APPEND CMAKE_LIBRARY_PATH /lib32)
    else()
      list(APPEND CMAKE_LIBRARY_PATH /lib)
    endif()
    if(EXISTS /usr/lib32)
      list(APPEND CMAKE_LIBRARY_PATH /usr/lib32)
    else()
      list(APPEND CMAKE_LIBRARY_PATH /usr/lib)
    endif()
  endif()
endif()

# Add these standard paths to the search paths for FIND_PATH
# to find include files from these locations first
if(MINGW)
  if(EXISTS /mingw)
      list(APPEND CMAKE_INCLUDE_PATH /mingw)
  endif()
  if(EXISTS /mingw32)
      list(APPEND CMAKE_INCLUDE_PATH /mingw32)
  endif()
  if(EXISTS /mingw64)
      list(APPEND CMAKE_INCLUDE_PATH /mingw64)
  endif()
endif()


option(ADORAD_NO_TEST "Do not build Adorad test binaries" OFF)
option(BUILD_BINARY "Build the C++ binaries" OFF)
option(BUILD_DOCS "Build Adorad documentation" OFF)
option(BUILD_PYTHON "Build Python binaries" ON)
option(USE_CUDA "Use CUDA" OFF) # TODO (turn ON)
option(USE_FAST_NVCC "Use parallel NVCC build" OFF)


# ------ CMake build directories ----
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


enable_testing()

#  compiler and linker options
#@      In case of Makefiles, if the user does not setup CMAKE_BUILD_TYPE, assume it's Release:
if(CMAKE_GENERATOR MATCHES "Makefiles|Ninja" AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "Build type not set - defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build from: Debug Release." FORCE)
endif()


# ---[ Main build
add_subdirectory(adorad)


file(GLOB SOURCES "adorad/*.cpp")
# Adding the executable 
add_executable(adorad main.cpp)

target_include_directories(
    adorad
    PRIVATE
    ${PROJECT_SOURCE_DIR}/adorad/coreten
)
