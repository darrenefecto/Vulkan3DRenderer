@echo off
setlocal
if not defined VULKAN_SDK (
    echo VULKAN_SDK is not set.
    exit /b 1
)
set "GLSLC=%VULKAN_SDK%\Bin\glslc.exe"
if not exist "%GLSLC%" (
    echo glslc.exe was not found: "%GLSLC%"
    exit /b 1
)
"%GLSLC%" "%~dp0custom.vert" -o "%~dp0custom.vert.spv" || exit /b 1
"%GLSLC%" "%~dp0custom.frag" -o "%~dp0custom.frag.spv" || exit /b 1
echo Custom shaders compiled.
