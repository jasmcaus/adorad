cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

project(ad CXX)

set(CMAKE_INSTALL_MESSAGE NEVER)

# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)
if(env_cxx_standard GREATER -1)
    message(
        WARNING "C++ standard version definition detected in environment variable."
        "Adorad requires -std=c++17. Please remove -std=c++ settings in your environment.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 11)


option(ADORAD_NO_TEST "Do not build ATen test binaries" OFF)
option(BUILD_BINARY "Build the C++ binaries" OFF)
option(BUILD_DOCS "Build Adorad documentation" OFF)
option(BUILD_PYTHON "Build Python binaries" ON)
option(USE_CUDA "Use CUDA" ON)
option(USE_FAST_NVCC "Use parallel NVCC build" OFF)

# ------ CMake build directories ----
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

enable_testing()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not set - defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build from: Debug Release." FORCE)
endif()


# ---[ Build flags
if(NOT MSVC)
    string(APPEND CMAKE_CXX_FLAGS " -O2 -fPIC")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-narrowing")
    string(APPEND CMAKE_CXX_FLAGS " -Wall")
    string(APPEND CMAKE_CXX_FLAGS " -Wextra")
    string(APPEND CMAKE_CXX_FLAGS " -Werror=return-type")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-missing-field-initializers")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-type-limits")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-array-bounds")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unknown-pragmas")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-sign-compare")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-parameter")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-variable")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-function")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-result")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-local-typedefs")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-strict-overflow")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-strict-aliasing")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-error=deprecated-declarations")

    if(CMAKE_COMPILER_IS_GNUCXX AND NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0.0))
        string(APPEND CMAKE_CXX_FLAGS " -Wno-stringop-overflow")
    endif()

    if(CMAKE_COMPILER_IS_GNUCXX)
        # Suppress "The ABI for passing parameters with 64-byte alignment has changed in GCC 4.6"
        string(APPEND CMAKE_CXX_FLAGS " -Wno-psabi")
    endif()

    string(APPEND CMAKE_CXX_FLAGS " -Wno-error=pedantic")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-error=redundant-decls")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-error=old-style-cast")

    # These flags are not available in GCC-4.8.5. Set only when using clang.
    # Compared against https://gcc.gnu.org/onlinedocs/gcc-4.8.5/gcc/Option-Summary.html
    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        string(APPEND CMAKE_CXX_FLAGS " -Wno-invalid-partial-specialization")
        string(APPEND CMAKE_CXX_FLAGS " -Wno-typedef-redefinition")
        string(APPEND CMAKE_CXX_FLAGS " -Wno-unknown-warning-option")
        string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-private-field")
        string(APPEND CMAKE_CXX_FLAGS " -Wno-inconsistent-missing-override")
        string(APPEND CMAKE_CXX_FLAGS " -Wno-aligned-allocation-unavailable")
        string(APPEND CMAKE_CXX_FLAGS " -Wno-c++14-extensions")
        string(APPEND CMAKE_CXX_FLAGS " -Wno-constexpr-not-const")
        string(APPEND CMAKE_CXX_FLAGS " -Wno-missing-braces")
        string(APPEND CMAKE_CXX_FLAGS " -Qunused-arguments")
        if(${COLORIZE_OUTPUT})
        string(APPEND CMAKE_CXX_FLAGS " -fcolor-diagnostics")
        endif()
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
        if(${COLORIZE_OUTPUT})
        string(APPEND CMAKE_CXX_FLAGS " -fdiagnostics-color=always")
        endif()
    endif()

    if((APPLE AND (NOT ("${CLANG_VERSION_STRING}" VERSION_LESS "9.0")))
        OR(CMAKE_COMPILER_IS_GNUCXX
        AND(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0 AND NOT APPLE)))
        string(APPEND CMAKE_CXX_FLAGS " -faligned-new")
    endif()

    if(WERROR)
        check_cxx_compiler_flag("-Werror" COMPILER_SUPPORT_WERROR)
        if(NOT COMPILER_SUPPORT_WERROR)
        set(WERROR FALSE)
        else()
        string(APPEND CMAKE_CXX_FLAGS " -Werror")
        endif()
    endif(WERROR)

    if(NOT APPLE)
        string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-but-set-variable")
        string(APPEND CMAKE_CXX_FLAGS " -Wno-maybe-uninitialized")
    endif()

    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fno-omit-frame-pointer -O0")
    string(APPEND CMAKE_LINKER_FLAGS_DEBUG " -fno-omit-frame-pointer -O0")
    string(APPEND CMAKE_CXX_FLAGS " -fno-math-errno")
    string(APPEND CMAKE_CXX_FLAGS " -fno-trapping-math")
    check_cxx_compiler_flag("-Werror=format" HAS_WERROR_FORMAT)

    if(HAS_WERROR_FORMAT)
        string(APPEND CMAKE_CXX_FLAGS " -Werror=format")
    endif()

    check_cxx_compiler_flag("-Werror=cast-function-type" HAS_WERROR_CAST_FUNCTION_TYPE)

    if(HAS_WERROR_CAST_FUNCTION_TYPE)
        string(APPEND CMAKE_CXX_FLAGS " -Werror=cast-function-type")
    endif()
endif()

# ---[ Main build
add_subdirectory(adorad)


file(GLOB SOURCES "adorad/*.cpp")
# Adding the executable 
add_executable(ad main.cpp)

target_include_directories(ad
    PRIVATE
        ${PROJECT_SOURCE_DIR}/adorad/coreten
)
