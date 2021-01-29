@ECHO OFF
@echo off 

pushd %~dp0

if "%1" == "" goto all
if "%1" == "all" goto all
if "%1" == "compile" goto compile
if "%1" == "run" goto run
if "%1" == "clean" goto clean


set COMPILER=g++
set TARGET=ad
set CPPSOURCE=main.cpp src/*.cpp
set HSOURCE=./src
REM compiler flags:
REM  -g     - adds debugging information to the executable file
REM  -Wall  - used to turn on most compiler warnings
set CFLAGS=-g -Wall

:all
@REM %COMPILER% %CFLAGS% %CPPSOURCE% -o %TARGET% -I %HSOURCE%
g++ -g -Wall main.cpp src/*.cpp -o ad -I ./src 
goto run 

:compile 
g++ -g -Wall main.cpp src/*.cpp -o ad -I ./src
@REM %COMPILER% %CFLAGS% %CPPSOURCE% -o %TARGET% -I %HSOURCE%
goto end

:run 
@REM %TARGET%
ad 
goto end 

:clean 
del %TARGET%.exe

@REM For non-windows users, use the following:
@REM rm %TARGET%.exe

goto end 

:help
@echo "Help"

:end 
popd