@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Vulkan Shader Compiler

set "GLSLC=C:\VulkanSDK\1.4.363.0\Bin\glslc.exe"
set "SOURCE=%~dp0"
set "OUTPUT=%~dp0Compiled"

echo ========================================
echo       Vulkan Shader Compiler
echo ========================================
echo.

if not exist "%GLSLC%" (
    echo ERROR: glslc.exe not found!
    pause
    exit /b 1
)

if not exist "%OUTPUT%" mkdir "%OUTPUT%"

for %%F in ("%SOURCE%*") do (
    if exist "%%~fF" (
        if not "%%~nxF"=="Compiler.bat" (
            call :ProcessFile "%%~fF"
        )
    )
)

echo.
echo ========================================
echo Finished! Files are in Compiled\
echo ========================================
pause
exit /b


:ProcessFile
set "FILE=%~1"
set "EXT=%~x1"
set "NAME=%~n1"

if /i "%EXT%"==".vert" goto Compile
if /i "%EXT%"==".frag" goto Compile
if /i "%EXT%"==".comp" goto Compile
if /i "%EXT%"==".geom" goto Compile
if /i "%EXT%"==".tesc" goto Compile
if /i "%EXT%"==".tese" goto Compile
if /i "%EXT%"==".mesh" goto Compile
if /i "%EXT%"==".task" goto Compile
if /i "%EXT%"==".rgen" goto Compile
if /i "%EXT%"==".rmiss" goto Compile
if /i "%EXT%"==".rchit" goto Compile
if /i "%EXT%"==".rahit" goto Compile
if /i "%EXT%"==".rint" goto Compile
if /i "%EXT%"==".rcall" goto Compile

copy /y "%FILE%" "%OUTPUT%\" >nul
echo [COPY] %~nx1
exit /b

:Compile
echo [COMPILE] %~nx1
"%GLSLC%" "%FILE%" -o "%OUTPUT%\%~nx1.spv"

if errorlevel 1 (
    echo [ERROR] %~nx1
    copy /y "%FILE%" "%OUTPUT%\" >nul
) else (
    echo [OK] %~nx1.spv
)
exit /b