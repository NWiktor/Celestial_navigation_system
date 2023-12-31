ECHO OFF
chcp 65001
cls

rem g++ may be used for only C++ handling of files
rem gcc

rem when compiler succeds runs .exe
g++ main.cpp ^
celestial_objects.cpp ^
-o main.exe && ^
main.exe

PAUSE
