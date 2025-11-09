# Deploy OTA Update Skill

## Metadata
- **Name**: deploy-ota
- **Description**: OTA 업데이트 패키지 생성 및 Fleet AI 플랫폼 배포
- **Phase**: Phase 7
- **Dependencies**: Android SDK, curl, zip
- **Estimated Time**: 5-10 minutes

## What This Skill Does

### 1. OTA Package Generation
- Release APK 서명
- Update payload 생성
- Metadata 파일 생성 (버전, 체크섬, 타겟 디바이스)

### 2. Package Upload
- Fleet AI 플랫폼 업로드
- 배포 타겟 설정 (Canary, Staged, Full)
- 롤백 지원

### 3. Deployment Verification
- 업로드 확인
- 체크섬 검증
- 배포 상태 모니터링

## Usage

```bash
# OTA 패키지 생성
./.claude/skills/deploy-ota/run.sh create --version 1.2.0

# 패키지 업로드 (Canary - 10% 디바이스)
./.claude/skills/deploy-ota/run.sh upload --version 1.2.0 --target canary

# 전체 배포
./.claude/skills/deploy-ota/run.sh deploy --version 1.2.0 --target all

# 롤백
./.claude/skills/deploy-ota/run.sh rollback --version 1.1.0
```

## Files Created
- `ota-packages/dtg_v1.2.0_ota.zip`
- `ota-packages/metadata.json`
- `ota-packages/checksums.sha256`
