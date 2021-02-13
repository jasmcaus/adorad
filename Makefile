default_target: all
.PHONY: default_target


# The CMake executable.
# CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"
CMAKE_COMMAND = cmake

TARGET = adorad
SOURCE_DIR = .
BUILD_DIR = build
GENERATOR = "MinGW Makefiles" 

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s
#Suppress display of executed commands.
$(VERBOSE).SILENT:


# PROCESS:
## 1. CMake Exec (generate the build files)
## 2. Run CMake Makefile
## 3. Run CXX compiled executable


all:
	cmake -s $(SOURCE_DIR) -B $(BUILD_DIR) -G $(GENERATOR)
	cd $(BUILD_DIR) && $(MAKE) && \
	echo --------------------------------------------
	echo --------------------------------------------
	cd $(BUILD_DIR) && $(TARGET)
.PHONY: all 


# Generate the CMake MinGW Makefiles
exec: 
	cmake -s $(SOURCE_DIR) -B $(BUILD_DIR) -G $(GENERATOR)
.PHONY: exec


make:
	cd $(BUILD_DIR) && $(MAKE)
.PHONY: make


run:
	cd $(BUILD_DIR) && $(TARGET)
.PHONY: run


clean: 
	@rmdir /Q /s $(BUILD_DIR) && mkdir $(BUILD_DIR)
.PHONY: clean


# ------------Minor Testing only -----------
COMPILER = g++
CPPSOURCE = test.cpp
# compiler flags:
#  -g     - adds debugging information to the executable file
# -Wall  - used to turn on most compiler warnings
CFLAGS = -g -Wall

test: 
	g++ $(CFLAGS) $(CPPSOURCE) -o test && test 
.PHONY: test