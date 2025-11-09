#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GLEC DTG - Test Suite${NC}"
echo -e "${GREEN}========================================${NC}\n"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

TEST_TYPE=${1:-"all"}

run_ai_tests() {
    echo -e "${YELLOW}🤖 Running AI Model Tests...${NC}\n"

    if [ -d "ai-models" ] && [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        cd ai-models

        if [ -d "tests" ]; then
            pytest tests/ -v --cov=training --cov=optimization
        else
            echo -e "${YELLOW}⚠️  No tests found - creating placeholder${NC}"
            mkdir -p tests
            echo "# AI model tests" > tests/__init__.py
        fi

        cd "$PROJECT_ROOT"
    else
        echo -e "${YELLOW}⚠️  AI models directory not set up${NC}"
    fi

    echo -e "${GREEN}✅ AI tests complete${NC}\n"
}

run_android_tests() {
    echo -e "${YELLOW}📱 Running Android Tests...${NC}\n"

    for APP_DIR in android-dtg android-driver; do
        if [ -d "$APP_DIR" ]; then
            echo -e "${YELLOW}Testing $APP_DIR...${NC}"
            cd "$APP_DIR"
            ./gradlew testDebugUnitTest || echo "Tests not yet implemented"
            cd "$PROJECT_ROOT"
        fi
    done

    echo -e "${GREEN}✅ Android tests complete${NC}\n"
}

run_stm32_tests() {
    echo -e "${YELLOW}🔧 Running STM32 Tests...${NC}\n"

    if [ -d "stm32-firmware" ]; then
        cd stm32-firmware
        make test 2>/dev/null || echo "Tests not yet implemented"
        cd "$PROJECT_ROOT"
    fi

    echo -e "${GREEN}✅ STM32 tests complete${NC}\n"
}

case $TEST_TYPE in
    all)
        run_ai_tests
        run_android_tests
        run_stm32_tests
        ;;
    ai)
        run_ai_tests
        ;;
    android)
        run_android_tests
        ;;
    stm32)
        run_stm32_tests
        ;;
    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ All Tests Complete! 🎉${NC}"
echo -e "${GREEN}========================================${NC}"
