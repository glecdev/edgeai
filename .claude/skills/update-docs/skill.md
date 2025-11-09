# Update Documentation Skill

## Metadata
- **Name**: update-docs
- **Description**: 코드-문서 자동 동기화 (Phase 6: Document)
- **Phase**: Phase 6 - Document
- **Dependencies**: pdoc, dokka, plantuml
- **Estimated Time**: 2-5 minutes

## What This Skill Does

### 1. API Documentation Generation
- Python: pdoc3 → HTML/Markdown
- Android: Dokka → HTML
- Auto-extract from docstrings

### 2. Architecture Diagrams Update
- PlantUML diagram generation
- Mermaid diagram generation
- Auto-sync with code structure

### 3. CLAUDE.md Synchronization
- Update component list
- Refresh command examples
- Sync workflow documentation

### 4. Changelog Generation
- Parse git commits (Conventional Commits)
- Auto-generate CHANGELOG.md
- Version bumping

## Usage

```bash
# Update API docs
./.claude/skills/update-docs/run.sh --api

# Update CLAUDE.md
./.claude/skills/update-docs/run.sh --claude

# Update all documentation
./.claude/skills/update-docs/run.sh --all

# Generate changelog
./.claude/skills/update-docs/run.sh --changelog
```
