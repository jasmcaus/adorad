set(the_description "The Core Functionality of Adorad")

cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

project(coreten CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Main build file for the Coreten library.
#
# Note that the Coreten library should maintain minimal dependencies - especially, it should not depend on any library that is implementation-specific or backend-specific. 


# Note: if you want to add ANY dependency to the coreten library, make sure you check with the core Adorad developers as 
# the dependency will be transitively passed on to all libraries dependent on Adorad.
file(GLOB CORETEN_SOURCES
        *.cpp
        core/*.cpp
        macros/*.cpp
        String/*.cpp
    )

file(GLOB_RECURSE CORETEN_TEST_FILES testing/*.cpp)
file(GLOB_RECURSE CORETEN_HEADERS *.h)

add_library(coreten $(CORETEN_SOURCES) $(CORETEN_HEADERS))

# Enable hidden visibility if compiler supports it.
if(${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
    target_compile_options(coreten PRIVATE "-fvisibility=hidden")
endif()


target_include_directories(
    coreten PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../> # maybe update this 
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)

if(USE_CUDA)
    add_subdirectory(cuda)
endif()


add_subdirectory(testing)


# ------Installation--------

# Note: for now, we will put all export path into one single Caffe2Targets group
# to deal with the cmake deployment need. Inside the Caffe2Targets set, the
# individual libraries like libc10.so and libcaffe2.so are still self-contained.
# install(TARGETS c10 EXPORT Caffe2Targets DESTINATION lib)
install(TARGETS coreten DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h>
        )

if(MSVC)
  install(FILES $<TARGET_PDB_FILE:coreten> DESTINATION lib OPTIONAL)
endif()