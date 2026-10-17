@echo off
setlocal

set "GLSLC=C:\VulkanSDK\1.4.363.0\Bin\glslc.exe"
set "SHADER_DIR=%~dp0"
set "OUTPUT_DIR=%~dp0.."

echo Vulkan Shader Compiler
echo Source: %SHADER_DIR%
echo Output: %OUTPUT_DIR%
echo.

if not exist "%GLSLC%" (
    echo [ERROR] glslc.exe was not found: %GLSLC%
    pause
    exit /b 1
)

REM Compile vertex shaders.
for %%F in ("%SHADER_DIR%*.vert") do (
    echo Compiling %%~nxF
    "%GLSLC%" -I "%SHADER_DIR%" "%%~fF" -o "%OUTPUT_DIR%\%%~nxF.spv"
    if errorlevel 1 (
        echo [ERROR] Failed to compile %%~nxF.
        pause
        exit /b 1
    )
)

REM Compile fragment shaders.
for %%F in ("%SHADER_DIR%*.frag") do (
    echo Compiling %%~nxF
    "%GLSLC%" -I "%SHADER_DIR%" "%%~fF" -o "%OUTPUT_DIR%\%%~nxF.spv"
    if errorlevel 1 (
        echo [ERROR] Failed to compile %%~nxF.
        pause
        exit /b 1
    )
)

REM Compile compute shaders.
for %%F in ("%SHADER_DIR%*.comp") do (
    echo Compiling %%~nxF
    "%GLSLC%" -I "%SHADER_DIR%" "%%~fF" -o "%OUTPUT_DIR%\%%~nxF.spv"
    if errorlevel 1 (
        echo [ERROR] Failed to compile %%~nxF.
        pause
        exit /b 1
    )
)

echo.
echo Shader compilation completed successfully.
pause
