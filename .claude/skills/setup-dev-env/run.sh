#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GLEC DTG Edge AI SDK - Dev Environment Setup${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${YELLOW}📍 Project Directory: $PROJECT_ROOT${NC}\n"

# Step 1: Check Python version
echo -e "${YELLOW}Step 1: Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ] && [ "$PYTHON_MINOR" -le 10 ]; then
    echo -e "${GREEN}✅ Python $PYTHON_VERSION detected (compatible)${NC}\n"
else
    echo -e "${RED}❌ Python 3.9 or 3.10 required (current: $PYTHON_VERSION)${NC}"
    echo "Please install Python 3.9 or 3.10 first"
    exit 1
fi

# Step 2: Create Python virtual environment
echo -e "${YELLOW}Step 2: Creating Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created: venv/${NC}\n"
else
    echo -e "${GREEN}✅ Virtual environment already exists: venv/${NC}\n"
fi

# Step 3: Activate virtual environment
echo -e "${YELLOW}Step 3: Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}\n"

# Step 4: Upgrade pip
echo -e "${YELLOW}Step 4: Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}✅ pip upgraded to $(pip --version | awk '{print $2}')${NC}\n"

# Step 5: Install dependencies
echo -e "${YELLOW}Step 5: Installing dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Dependencies installed from requirements.txt${NC}\n"
else
    echo "requirements.txt not found, installing core packages..."
    pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install tensorflow==2.14.0
    pip install onnx==1.15.0 onnx2tf==1.17.5
    pip install lightgbm==4.1.0
    pip install scikit-learn==1.3.2
    pip install tsaug==0.2.1
    pip install mlflow==2.9.0
    pip install dvc==3.35.0
    pip install pytest==7.4.3 pytest-cov==4.1.0
    echo -e "${GREEN}✅ Core AI packages installed${NC}\n"
fi

# Step 6: Check Docker
echo -e "${YELLOW}Step 6: Checking Docker installation...${NC}"
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | awk '{print $3}' | tr -d ',')
    echo -e "${GREEN}✅ Docker $DOCKER_VERSION detected${NC}"

    if command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version | awk '{print $4}' | tr -d ',')
        echo -e "${GREEN}✅ Docker Compose $COMPOSE_VERSION detected${NC}\n"
    else
        echo -e "${YELLOW}⚠️  Docker Compose not found (optional)${NC}\n"
    fi
else
    echo -e "${YELLOW}⚠️  Docker not found (optional for local development)${NC}"
    echo "Install from: https://docs.docker.com/get-docker/"
    echo ""
fi

# Step 7: Initialize Git (if not already)
echo -e "${YELLOW}Step 7: Initializing Git...${NC}"
if [ ! -d ".git" ]; then
    git init
    echo -e "${GREEN}✅ Git repository initialized${NC}\n"
else
    echo -e "${GREEN}✅ Git repository already exists${NC}\n"
fi

# Step 8: Create .gitignore
echo -e "${YELLOW}Step 8: Creating .gitignore...${NC}"
if [ ! -f ".gitignore" ]; then
    cat > .gitignore << 'EOF'
# Python
venv/
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# PyCharm
.idea/

# VSCode
.vscode/

# AI Models
*.pth
*.pt
*.onnx
*.dlc
*.tflite
*.pb
mlruns/

# Data
data/raw/
data/processed/
*.csv
*.parquet
!data/sample/

# DVC
.dvc/cache
.dvc/tmp

# Android
*.apk
*.ap_
*.aab
*.dex
*.class
bin/
gen/
out/
.gradle/
local.properties
.idea/
*.iml
captures/
.externalNativeBuild/
.cxx/

# STM32
*.o
*.d
*.elf
*.hex
*.bin
*.list
*.map

# OS
.DS_Store
Thumbs.db

# Environment
.env
*.log
EOF
    echo -e "${GREEN}✅ .gitignore created${NC}\n"
else
    echo -e "${GREEN}✅ .gitignore already exists${NC}\n"
fi

# Step 9: Initialize DVC
echo -e "${YELLOW}Step 9: Initializing DVC...${NC}"
if [ ! -d ".dvc" ]; then
    dvc init
    echo -e "${GREEN}✅ DVC initialized${NC}\n"
else
    echo -e "${GREEN}✅ DVC already initialized${NC}\n"
fi

# Step 10: MLflow setup
echo -e "${YELLOW}Step 10: Setting up MLflow...${NC}"
mkdir -p mlruns
mkdir -p mlartifacts
echo -e "${GREEN}✅ MLflow directories created${NC}"
echo -e "To start MLflow server: ${YELLOW}mlflow server --host 0.0.0.0 --port 5000${NC}\n"

# Step 11: Create requirements.txt if it doesn't exist
echo -e "${YELLOW}Step 11: Creating requirements.txt...${NC}"
if [ ! -f "requirements.txt" ]; then
    pip freeze > requirements.txt
    echo -e "${GREEN}✅ requirements.txt created with $(wc -l < requirements.txt) packages${NC}\n"
else
    echo -e "${GREEN}✅ requirements.txt already exists${NC}\n"
fi

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Development Environment Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo -e "${YELLOW}📋 Summary:${NC}"
echo "  • Python: $PYTHON_VERSION"
echo "  • Virtual environment: venv/"
echo "  • Packages installed: $(pip list --format=freeze | wc -l)"
echo "  • Git initialized: Yes"
echo "  • DVC initialized: Yes"
echo ""

echo -e "${YELLOW}🚀 Next Steps:${NC}"
echo "  1. Activate virtual environment: ${GREEN}source venv/bin/activate${NC}"
echo "  2. Start MLflow server: ${GREEN}mlflow server --host 0.0.0.0 --port 5000${NC}"
echo "  3. Access MLflow UI: ${GREEN}http://localhost:5000${NC}"
echo "  4. Run tests: ${GREEN}pytest tests/${NC}"
echo ""

echo -e "${YELLOW}📚 Useful Commands:${NC}"
echo "  • Install new package: ${GREEN}pip install <package>${NC}"
echo "  • Update requirements: ${GREEN}pip freeze > requirements.txt${NC}"
echo "  • Track data with DVC: ${GREEN}dvc add data/dataset.csv${NC}"
echo "  • Log experiment: ${GREEN}mlflow run .${NC}"
echo ""

echo -e "${GREEN}Happy Coding! 🎉${NC}"
