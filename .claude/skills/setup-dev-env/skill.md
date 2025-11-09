# Setup Development Environment Skill

## Metadata
- **Name**: setup-dev-env
- **Description**: GLEC DTG Edge AI SDK 개발 환경 자동 설정
- **Phase**: Phase 1
- **Dependencies**: Python 3.9+, Docker, Git
- **Estimated Time**: 5-10 minutes

## What This Skill Does

### 1. Python Virtual Environment
- Python 3.9 또는 3.10 가상환경 생성
- 필수 AI 라이브러리 설치 (PyTorch, TensorFlow, ONNX)
- requirements.txt 기반 의존성 설치

### 2. Docker Setup
- Dockerfile 생성 (재현 가능한 개발 환경)
- docker-compose.yml 생성 (MLflow, PostgreSQL)
- 개발용 Docker 이미지 빌드

### 3. Git Configuration
- .gitignore 설정 (Python, Android, STM32)
- Git hooks 설정 (pre-commit)
- 브랜치 전략 초기화

### 4. DVC Initialization
- DVC 초기화 (데이터 버전 관리)
- 로컬 스토리지 설정
- .dvc 디렉토리 구성

### 5. MLflow Setup
- MLflow 추적 서버 시작
- 실험 저장소 설정
- 로컬 artifact 저장소 구성

## Usage

### From Command Line
```bash
cd /path/to/edgeai
./.claude/skills/setup-dev-env/run.sh
```

### From Claude Code
```
Please run the setup-dev-env skill to configure my development environment.
```

## Expected Output
```
✅ Python 가상환경 생성 완료 (venv/)
✅ 의존성 설치 완료 (requirements.txt)
✅ Dockerfile 생성 완료
✅ Git 초기화 완료
✅ DVC 초기화 완료
✅ MLflow 서버 시작 (http://localhost:5000)

🎉 개발 환경 설정이 완료되었습니다!

다음 단계:
1. 가상환경 활성화: source venv/bin/activate
2. Docker 이미지 빌드: docker-compose build
3. MLflow UI 접속: http://localhost:5000
```

## Files Created
- `venv/` - Python 가상환경
- `requirements.txt` - Python 의존성
- `Dockerfile` - Docker 이미지 설정
- `docker-compose.yml` - Docker Compose 설정
- `.gitignore` - Git 무시 파일
- `.pre-commit-config.yaml` - Pre-commit hooks
- `.dvc/` - DVC 설정
- `mlruns/` - MLflow 실험 결과

## Troubleshooting

### Python 버전 오류
```bash
# Python 3.9 또는 3.10 설치 확인
python3 --version

# pyenv 사용 (권장)
pyenv install 3.10.13
pyenv local 3.10.13
```

### Docker 권한 오류
```bash
# Docker 그룹에 사용자 추가
sudo usermod -aG docker $USER
newgrp docker
```

### MLflow 포트 충돌
```bash
# 다른 포트 사용 (예: 5001)
mlflow server --host 0.0.0.0 --port 5001
```
