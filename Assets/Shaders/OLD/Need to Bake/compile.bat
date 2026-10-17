@echo off
setlocal

echo ========================================
echo       Vulkan Shader Compiler
echo ========================================
echo.

REM Vulkan SDK
set "GLSLC=C:\VulkanSDK\1.4.363.0\Bin\glslc.exe"

REM Shader-Ordner = Ordner, in dem diese BAT liegt
set "SHADER_DIR=%~dp0"

echo glslc:
echo %GLSLC%
echo.
echo Shader folder:
echo %SHADER_DIR%
echo.

REM glslc prüfen
if not exist "%GLSLC%" (
    echo [ERROR] glslc.exe wurde nicht gefunden!
    echo.
    echo %GLSLC%
    echo.
    pause
    exit /b 1
)

echo [OK] glslc.exe gefunden!
echo.

REM Alle Vertex Shader
for %%F in ("%SHADER_DIR%*.vert") do (
    echo Compiling %%~nxF
    "%GLSLC%" "%%F" -o "%%F.spv"

    if errorlevel 1 (
        echo [ERROR] %%~nxF konnte nicht kompiliert werden!
        pause
        exit /b 1
    )
)

REM Alle Fragment Shader
for %%F in ("%SHADER_DIR%*.frag") do (
    echo Compiling %%~nxF
    "%GLSLC%" "%%F" -o "%%F.spv"

    if errorlevel 1 (
        echo [ERROR] %%~nxF konnte nicht kompiliert werden!
        pause
        exit /b 1
    )
)

REM Alle Compute Shader
for %%F in ("%SHADER_DIR%*.comp") do (
    echo Compiling %%~nxF
    "%GLSLC%" "%%F" -o "%%F.spv"

    if errorlevel 1 (
        echo [ERROR] %%~nxF konnte nicht kompiliert werden!
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo     ALLE SHADER ERFOLGREICH!
echo ========================================
echo.
pause