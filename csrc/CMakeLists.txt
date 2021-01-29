# from adorad import Tensor as tens 

# x = tens.eye(3)
# y = tens([[1,2,3]])
# z = y.matmul(x).sum()
# z.backward()

# print(x.grad)
# print(y.grad)


#
# This is a CMake makefile.  CMake is a tool that helps you build C++ programs.
# You can download CMake from http://www.cmake.org.  This CMakeLists.txt file
# builds adorad's internal testing program.
#


cmake_minimum_required(VERSION 3.7)

# Every project needs a name.
project(ad)


add_library(Adorad ./adorad)
target_include_directories(Adorad PUBLIC ./adorad)


# Adding the executable 
add_executable(adorad main.cpp)
target_link_libraries(ad Adorad)



# # adder use logic
# option(USE_ADORAD "Use Adorad?." ON)
# option(USE_CORETEN "Use Coreten?." ON)

# if(USE_ADORAD)
# 	add_subdirectory(adorad)

# 	list(APPEND EXTRA_LIB_DIRS "adorad")
# 	list(APPEND EXTRA_INCLUDE_DIRS "adorad")
# 	list(APPEND EXTRA_LINKS adder)

# endif()

# if(USE_CORETEN)
# 	add_subdirectory(coreten)

# 	list(APPEND EXTRA_LIB_DIRS "coreten")
# 	list(APPEND EXTRA_INCLUDE_DIRS "coreten")
# 	list(APPEND EXTRA_LINKS adder)

# endif()


# Tell cmake we will need "Adorad" and "Coreten".  This command will pull in these folders and compile it
# into the project.  Note that you don't need to compile or install anything.  All
# cmake needs are the source code folders and it will take care of everything.
# add_subdirectory(adorad)
# add_subdirectory(coreten)



# add_library(
#     adorad 
#     adorad/src/Tensor.h
#     adorad/src/TensorFuncs.h
#     adorad/src/m.cpp
# )

# The next thing we need to do is tell CMake about the code you want to
# compile.  We do this with the add_executable() statement which takes the name
# of the output executable and then a list of .cpp files to compile.  If your 
# program consisted of multiple .cpp files you would simply list them here in 
# the add_executable() statement.  
# add_executable(${PROJECT_NAME} main.cpp)

# Finally, you need to tell CMake that this program depends on dlib.  You do 
# that with this statement: 
# target_link_libraries(assignment_learning_ex dlib::dlib)
# target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_BINARY_DIR})