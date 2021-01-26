@ECHO OFF

pushd %~dp0

if "%1" == "" goto all
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


:alll
@echo "All"
%COMPILER% %CFLAGS% %CPPSOURCE% -o %TARGET% -I %HSOURCE%
%TARGET%
goto end 

:compile 
@echo "Compile"
%COMPILER% %CFLAGS% %CPPSOURCE% -o %TARGET% -I %HSOURCE%
goto end 

:run 
@echo "Run"
%TARGET%
goto end 

:clean 
@echo "clean"
del %TARGET%.exe

@REM For non-windows users, uncomment the following:
@REM rm %TARGET%.exe

goto end 

:help
@echo "Help"


:end
popd