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

set VAR=g++ bi
set VAR1=g++ djkndj djd djd id
set VAR2=g++ djkndj djd djd id
set VAR3=g++ djkndj djd djd id
set VAR4=g++ djkndj djd djd id
set VAR5=g++ djkndj djd djd id
set VAR6=g++ djkndj djd djd id
set VAR8=g+ djkndj djd djd id

:all
@echo "%COMPILER% %CFLAGS% %CPPSOURCE% -o %TARGET% -I %HSOURCE%"
goto end 

:compile 
g++ -g -Wall main.cpp src/*.cpp -o ad -I ./src
@REM %COMPILER% %CFLAGS% %CPPSOURCE% -o %TARGET% -I %HSOURCE%
goto end

:run 
%TARGET%
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