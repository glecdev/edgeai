@echo off
echo ========================================
echo Building Android DTG with Android Studio Java
echo ========================================
echo.

set "JAVA_HOME=C:\Program Files\Android\Android Studio\jbr"
set "PATH=%JAVA_HOME%\bin;%PATH%"

echo Using Java from: %JAVA_HOME%
echo.

java -version
echo.

echo Starting Gradle build...
echo.

cd /d "%~dp0"
call gradlew.bat clean assembleDebug

echo.
echo ========================================
echo Build complete!
echo ========================================
echo.

if exist "app\build\outputs\apk\debug\app-debug.apk" (
    echo SUCCESS: APK generated at:
    echo app\build\outputs\apk\debug\app-debug.apk
    dir "app\build\outputs\apk\debug\app-debug.apk"
) else (
    echo WARNING: APK not found
)

pause
