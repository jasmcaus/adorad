cmake_minimum_required(VERSION 3.10.1)
project (TestLibrary)

#For the shared library:
set ( PROJECT_LINK_LIBS libad.dll )
link_directories( F:/Documents/Code Projects/PyPi/adorad/cmake/build2 )

#For the static library:
#set ( PROJECT_LINK_LIBS libtestStudent.a )
#link_directories( F:/Documents/Code Projects/PyPi/adorad/cmake/build2 )

include_directories(include)

add_executable(libtest math.cpp)
target_link_libraries(libtest ${PROJECT_LINK_LIBS} )