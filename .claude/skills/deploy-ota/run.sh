#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GLEC DTG - OTA Deployment${NC}"
echo -e "${GREEN}========================================${NC}\n"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

ACTION=${1:-"create"}
VERSION=""
TARGET="canary"

shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --version|-v)
            VERSION="$2"
            shift 2
            ;;
        --target|-t)
            TARGET="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [ -z "$VERSION" ]; then
    echo -e "${RED}❌ Version required: --version 1.2.0${NC}"
    exit 1
fi

OTA_DIR="$PROJECT_ROOT/ota-packages"
mkdir -p "$OTA_DIR"

echo -e "${YELLOW}📦 OTA Deployment${NC}"
echo "  • Action: $ACTION"
echo "  • Version: $VERSION"
echo "  • Target: $TARGET"
echo ""

create_ota_package() {
    echo -e "${YELLOW}📦 Creating OTA package for v$VERSION...${NC}\n"

    # Find release APK
    APK_PATH="$PROJECT_ROOT/android-dtg/app/build/outputs/apk/release/app-release.apk"

    if [ ! -f "$APK_PATH" ]; then
        echo -e "${RED}❌ Release APK not found: $APK_PATH${NC}"
        echo "Build release first: ./gradlew assembleRelease"
        exit 1
    fi

    # Create metadata
    cat > "$OTA_DIR/metadata.json" << EOF
{
  "version": "$VERSION",
  "build_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "min_version": "1.0.0",
  "target_devices": ["dtg_snapdragon_865", "dtg_snapdragon_888"],
  "changelog": [
    "Performance improvements",
    "Bug fixes",
    "Security updates"
  ]
}
EOF

    # Create OTA package
    OTA_PACKAGE="$OTA_DIR/dtg_v${VERSION}_ota.zip"
    zip -j "$OTA_PACKAGE" "$APK_PATH" "$OTA_DIR/metadata.json"

    # Generate checksums
    sha256sum "$OTA_PACKAGE" > "$OTA_DIR/checksums.sha256"

    echo -e "${GREEN}✅ OTA Package Created!${NC}"
    echo "  • Package: $OTA_PACKAGE"
    echo "  • Size: $(stat -f%z "$OTA_PACKAGE" 2>/dev/null || stat -c%s "$OTA_PACKAGE") bytes"
    echo ""
}

upload_package() {
    echo -e "${YELLOW}📤 Uploading to Fleet AI Platform...${NC}\n"

    OTA_PACKAGE="$OTA_DIR/dtg_v${VERSION}_ota.zip"

    if [ ! -f "$OTA_PACKAGE" ]; then
        echo -e "${RED}❌ OTA package not found${NC}"
        create_ota_package
    fi

    # Simulated upload (replace with actual API endpoint)
    UPLOAD_URL="https://api.glec.ai/ota/upload"

    echo -e "${YELLOW}Uploading to $UPLOAD_URL...${NC}"
    echo -e "${GREEN}✅ Upload successful (simulated)${NC}"
    echo "  • Target: $TARGET"
    echo "  • Rollout: $([ "$TARGET" == "canary" ] && echo "10%" || echo "100%")"
    echo ""
}

case $ACTION in
    create)
        create_ota_package
        ;;
    upload)
        create_ota_package
        upload_package
        ;;
    deploy)
        create_ota_package
        upload_package
        echo -e "${GREEN}✅ Deployment initiated!${NC}"
        ;;
    *)
        echo -e "${RED}Unknown action: $ACTION${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}✅ OTA Process Complete! 🚀${NC}"
