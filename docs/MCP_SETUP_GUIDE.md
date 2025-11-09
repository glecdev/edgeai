# MCP 서버 설정 가이드
## GLEC DTG Edge AI SDK 프로젝트

### 📍 Claude Desktop 설정 파일 위치

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

---

## 🚀 Step-by-Step 설정 방법

### Step 1: Claude Desktop 종료
Claude Desktop 앱을 완전히 종료합니다.

### Step 2: 설정 파일 편집

#### macOS 사용자
```bash
# 디렉토리 생성 (없는 경우)
mkdir -p ~/Library/Application\ Support/Claude

# 설정 파일 편집 (VSCode 사용 예시)
code ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

#### Linux 사용자
```bash
# 디렉토리 생성 (없는 경우)
mkdir -p ~/.config/Claude

# 설정 파일 편집
nano ~/.config/Claude/claude_desktop_config.json
```

### Step 3: 필수 MCP 서버 설정 추가

**기존 파일이 없는 경우** - 아래 내용 전체 복사:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/absolute/path/to/edgeai"
      ]
    },
    "git": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-git",
        "--repository",
        "/absolute/path/to/edgeai"
      ]
    },
    "memory": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-memory"
      ]
    }
  }
}
```

**기존 파일이 있는 경우** - `mcpServers` 객체에 추가:

```json
{
  "mcpServers": {
    // ... 기존 MCP 서버 설정 ...
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/absolute/path/to/edgeai"]
    },
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git", "--repository", "/absolute/path/to/edgeai"]
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

### Step 4: 경로 수정

`/absolute/path/to/edgeai`를 실제 프로젝트 경로로 변경하세요:

#### macOS 예시
```json
"/Users/yourname/Projects/edgeai"
```

#### Linux 예시
```json
"/home/yourname/edgeai"
```

#### Windows 예시
```json
"C:\\Users\\yourname\\Projects\\edgeai"
```

**현재 디렉토리 경로 확인 방법**:
```bash
cd /path/to/edgeai
pwd
```

### Step 5: 파일 저장 및 Claude Desktop 재시작

1. 설정 파일 저장 (Ctrl+S 또는 Cmd+S)
2. Claude Desktop 앱 재시작

### Step 6: MCP 서버 활성화 확인

Claude Desktop에서 새 대화 시작 후:

```
현재 사용 가능한 MCP 도구를 나열해주세요.
```

다음과 같은 도구들이 나타나야 합니다:
- `read_file` (filesystem MCP)
- `write_file` (filesystem MCP)
- `list_directory` (filesystem MCP)
- `git_status` (git MCP)
- `git_commit` (git MCP)
- `create_entities` (memory MCP)
- `read_graph` (memory MCP)

---

## 📊 각 MCP 서버 설명

### 1. filesystem MCP
**목적**: 파일 시스템 작업 간소화

**주요 기능**:
- 대량 파일 생성/수정
- 디렉토리 트리 구조화
- 파일 검색 및 필터링

**사용 예시** (Phase 1):
- 전체 프로젝트 디렉토리 구조 한 번에 생성
- 여러 README.md 파일 동시 생성

### 2. git MCP
**목적**: Git 작업 자동화

**주요 기능**:
- 스테이징 및 커밋
- 브랜치 생성/전환
- Git 상태 조회
- 변경사항 diff 확인

**사용 예시** (모든 Phase):
- 자동 커밋 메시지 생성
- 브랜치 관리 간소화

### 3. memory MCP
**목적**: 세션 간 컨텍스트 유지

**주요 기능**:
- 설계 결정사항 저장
- 하이퍼파라미터 기록
- 실험 결과 추적

**사용 예시** (Phase 2):
- AI 모델 학습 하이퍼파라미터 기억
- 최적 양자화 설정 저장
- 이전 실험 결과 참조

---

## 🔧 선택적 MCP 서버 (권장)

### 설정 파일에 추가 가능:

```json
{
  "mcpServers": {
    // ... 필수 MCP 3개 ...

    "fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"]
    },
    "sqlite": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-sqlite",
        "--db-path",
        "/absolute/path/to/edgeai/test.db"
      ]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
      }
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your_brave_api_key"
      }
    }
  }
}
```

**각 MCP 용도**:
- **fetch**: 외부 데이터셋 다운로드 (Phase 2)
- **sqlite**: 오프라인 큐 DB 설계 (Phase 5)
- **github**: 오픈소스 예제 검색 (Phase 4)
- **brave-search**: STM32 HAL 예제 검색 (Phase 3)

---

## ⚠️ 문제 해결

### 문제 1: MCP 서버가 나타나지 않음
**해결**:
1. Claude Desktop 완전 종료 (Cmd+Q 또는 Alt+F4)
2. 설정 파일 JSON 문법 검증: https://jsonlint.com
3. Claude Desktop 재시작

### 문제 2: "npx: command not found"
**해결**:
```bash
# Node.js 및 npm 설치
# macOS
brew install node

# Linux (Ubuntu/Debian)
sudo apt install nodejs npm

# 설치 확인
npx --version
```

### 문제 3: 경로 오류
**해결**:
- 절대 경로 사용 (상대 경로 ❌)
- Windows: 백슬래시 이스케이프 (`C:\\Users\\...`)
- 공백 포함 경로: 따옴표로 감싸기

### 문제 4: GitHub MCP 인증 오류
**해결**:
1. GitHub Personal Access Token 생성:
   - https://github.com/settings/tokens
   - 권한: `repo`, `read:org`
2. 토큰을 설정 파일에 추가

---

## 📋 설정 확인 체크리스트

- [ ] Claude Desktop 종료
- [ ] 설정 파일 생성/편집
- [ ] 필수 MCP 3개 추가 (filesystem, git, memory)
- [ ] 프로젝트 경로를 절대 경로로 수정
- [ ] JSON 문법 검증
- [ ] 파일 저장
- [ ] Claude Desktop 재시작
- [ ] 새 대화에서 MCP 도구 확인

---

## 🎯 다음 단계

MCP 설정 완료 후:
1. ✅ Custom Skills 구현 (`.claude/skills/` 디렉토리)
2. ✅ Phase 1 환경 설정 시작
3. ✅ 프로젝트 구조 자동 생성

**설정 완료 시 Claude Code에 알려주세요:**
```
MCP 서버 설정이 완료되었습니다. Custom Skills 구현을 진행해주세요.
```
