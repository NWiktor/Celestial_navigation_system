echo off
chcp 65001


:script
cls
dir *.py /s /b > output
echo **** Avalaiable files for PyLinter check ****
setlocal EnableDelayedExpansion
set n=1
for /F "tokens=*" %%a in (output) do (
	rem For %%A in ("%filename%") do (echo %%~nA)
	rem echo %%~nxa
	echo [!n!]  %%~nxa
	set vector[!n!]=%%a
	set /A n+=1
	)
echo [x]  Exit program
del output


:choice_select
set choice=retry
set /p "choice=Select option: "
if %choice%==retry (goto choice_select)
if %choice%==x (goto end)
goto run_file


:run_file
if not exist !vector[%choice%]! (
	echo Invalid selection!
	goto choice_select
	)
cls
pylint !vector[%choice%]! --output-format=colorized
set exit=run
set /p "exit=Press 'Enter' to restart, or 'x' to return: "
if %exit%==x (goto script)
goto run_file

:end