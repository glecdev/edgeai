#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GLEC DTG - Android Build${NC}"
echo -e "${GREEN}========================================${NC}\n"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

APP_TYPE=${1:-"dtg"}
BUILD_TYPE="Debug"
INSTALL=false
SHOW_LOG=false

# Parse arguments
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --release|-r)
            BUILD_TYPE="Release"
            shift
            ;;
        --install|-i)
            INSTALL=true
            shift
            ;;
        --log|-l)
            SHOW_LOG=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo -e "${YELLOW}📱 Building Android App${NC}"
echo "  • App: $APP_TYPE"
echo "  • Build Type: $BUILD_TYPE"
echo ""

# Build function
build_app() {
    local APP_DIR="$PROJECT_ROOT/android-$1"

    if [ ! -d "$APP_DIR" ]; then
        echo -e "${YELLOW}⚠️  Creating $1 app directory...${NC}"
        mkdir -p "$APP_DIR"
        echo -e "${GREEN}✅ Directory created: $APP_DIR${NC}\n"
        return
    fi

    cd "$APP_DIR"

    if [ ! -f "gradlew" ]; then
        echo -e "${YELLOW}⚠️  Gradle wrapper not found - initializing...${NC}"
        gradle wrapper
    fi

    echo -e "${YELLOW}🔨 Building $1 app ($BUILD_TYPE)...${NC}"

    if [ "$BUILD_TYPE" == "Release" ]; then
        ./gradlew assembleRelease
        APK_PATH="app/build/outputs/apk/release/app-release.apk"
    else
        ./gradlew assembleDebug
        APK_PATH="app/build/outputs/apk/debug/app-debug.apk"
    fi

    if [ -f "$APK_PATH" ]; then
        APK_SIZE=$(stat -f%z "$APK_PATH" 2>/dev/null || stat -c%s "$APK_PATH" 2>/dev/null)
        APK_SIZE_MB=$((APK_SIZE / 1024 / 1024))
        echo -e "${GREEN}✅ Build successful!${NC}"
        echo "  • APK: $APK_PATH"
        echo "  • Size: $APK_SIZE_MB MB"

        if [ "$INSTALL" = true ]; then
            echo -e "\n${YELLOW}📥 Installing to device...${NC}"
            adb install -r "$APK_PATH"
            echo -e "${GREEN}✅ Installed${NC}"
        fi
    fi
}

case $APP_TYPE in
    dtg)
        build_app "dtg"
        ;;
    driver)
        build_app "driver"
        ;;
    all)
        build_app "dtg"
        build_app "driver"
        ;;
    *)
        echo -e "${RED}Unknown app: $APP_TYPE${NC}"
        exit 1
        ;;
esac

if [ "$SHOW_LOG" = true ]; then
    echo -e "\n${YELLOW}📋 Starting logcat...${NC}"
    adb logcat -s DTGService:V AIInference:V BLE:V
fi

echo -e "\n${GREEN}✅ Android Build Complete! 🎉${NC}"
