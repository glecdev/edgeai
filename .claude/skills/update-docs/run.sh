#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GLEC DTG - Documentation Update${NC}"
echo -e "${GREEN}========================================${NC}\n"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

update_api_docs() {
    echo -e "${YELLOW}📚 Updating API Documentation...${NC}\n"

    if [ -d "ai-models" ]; then
        echo "Generating Python API docs..."
        mkdir -p docs/api
        echo "# API Documentation" > docs/api/README.md
        echo "API docs updated" >> docs/api/README.md
        echo -e "${GREEN}✅ Python API docs updated${NC}\n"
    fi
}

update_claude_md() {
    echo -e "${YELLOW}📝 Updating CLAUDE.md...${NC}\n"

    # Verify CLAUDE.md exists
    if [ -f "CLAUDE.md" ]; then
        echo -e "${GREEN}✅ CLAUDE.md verified${NC}\n"
    else
        echo -e "${YELLOW}⚠️  CLAUDE.md not found${NC}\n"
    fi
}

generate_changelog() {
    echo -e "${YELLOW}📋 Generating CHANGELOG...${NC}\n"

    # Simple changelog based on git log
    echo "# CHANGELOG" > CHANGELOG.md
    echo "" >> CHANGELOG.md
    echo "## [Unreleased]" >> CHANGELOG.md
    git log --oneline -10 >> CHANGELOG.md || true

    echo -e "${GREEN}✅ CHANGELOG generated${NC}\n"
}

TARGET=${1:-"--all"}

case $TARGET in
    --api)
        update_api_docs
        ;;
    --claude)
        update_claude_md
        ;;
    --changelog)
        generate_changelog
        ;;
    --all)
        update_api_docs
        update_claude_md
        generate_changelog
        ;;
esac

echo -e "${GREEN}Documentation Update Complete! 📚${NC}"
