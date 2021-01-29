#
# This is a CMake makefile.  CMake is a tool that helps you build C++ programs.
# You can download CMake from http://www.cmake.org.  This CMakeLists.txt file
# builds adorad's internal testing program.
#


cmake_minimum_required(VERSION 3.7)

# Every project needs a name.
project(adorad)

# Tell cmake we will need "Adorad" and "Coreten".  This command will pull in these folders and compile it
# into the project.  Note that you don't need to compile or install anything.  All
# cmake needs are the source code folders and it will take care of everything.
add_subdirectory(adorad)
add_subdirectory(coreten)



add_library(
    adorad 
    adorad/src/Tensor.h
    adorad/src/TensorFuncs.h
    adorad/src/m.cpp
)

# The next thing we need to do is tell CMake about the code you want to
# compile.  We do this with the add_executable() statement which takes the name
# of the output executable and then a list of .cpp files to compile.  If your 
# program consisted of multiple .cpp files you would simply list them here in 
# the add_executable() statement.  
add_excutable(adorad main.cpp)

# Finally, you need to tell CMake that this program depends on dlib.  You do 
# that with this statement: 
target_link_libraries(assignment_learning_ex dlib::dlib)