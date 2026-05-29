<h1 align="center">CoderMind</h1>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a> |
  <a href="README.ja-JP.md">日本語</a> |
  <a href="README.ko-KR.md">한국어</a> |
  <a href="README.hi-IN.md">हिन्दी</a>
</p>

## 코딩 에이전트가 편집하기 전에 계획을 세우게 하세요

코딩 에이전트는 로컬 편집에는 강하지만, 안정적인 계획 구조가 없으면 저장소 수준의 작업은 실패하기 쉽습니다. 요구사항이 흐트러지고, 아키텍처 결정이 사라지고, 여러 파일에 걸친 생성이 일관성을 잃으며, 업데이트가 숨겨진 의존성을 놓칠 수 있습니다.

CoderMind은 Claude Code와 GitHub Copilot에 저장소 수준의 코딩을 위한 **영속적인 RPG 워크스페이스**를 제공합니다. 이 워크스페이스는 요구사항, 기능, 아키텍처, 파일, 코드 엔티티, 의존성을 연결하는 **Repository Planning Graph (RPG)** 를 중심으로 구성되어 있습니다.

CoderMind을 사용하면 에이전트는 그래프 기반 워크플로로 작업할 수 있습니다:

- **Build (구축)**: 요구사항을 RPG 계획으로 바꾼 다음 여러 파일로 구성된 저장소를 생성합니다.
- **Understand (이해)**: 기존 저장소를 RPG로 매핑한 다음 검색, 탐색, 설명합니다.
- **Update (업데이트)**: 영향을 받는 RPG 노드를 식별하고, 편집 계획을 세우고, 코드와 그래프를 함께 업데이트합니다.

### 워크플로 선택

| 목표 | 워크플로 | 시작 위치 |
|---|---|---|
| 요구사항으로 새 저장소 구축 | Build 워크플로 (requirements → RPG → code) | [`Quick Start: 새 저장소`](#quick-start-새-저장소) |
| 기존 저장소 이해 | Understand 워크플로 (repository → RPG → search/explore) | [`Quick Start: 기존 저장소`](#quick-start-기존-저장소) |
| 기존 저장소 업데이트 | Update 워크플로 (change request → affected RPG nodes → edit plan → code/RPG update) | [`Quick Start: 기존 저장소`](#quick-start-기존-저장소) |

### 자세한 파이프라인

처음 사용하는 사용자는 이 섹션을 건너뛰고 아래의 Quick Start로 바로 시작할 수 있습니다.

<details>
<summary>커맨드 수준의 전체 워크플로 다이어그램</summary>

```text
Forward Direction: Requirements → RPG → Code

 Phase 1: Feature Specification       Phase 2: RPG Construction & Planning                             Phase 3
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ feature  │ │ feature  │ │ feature  │ │  build   │ │  build   │ │ design   │ │ design   │ │  plan    │ │          │
│  _spec   ├─▶  _build  ├─▶_refactor ├─▶ skeleton ├─▶  data    ├─▶  base    ├─▶interfaces├─▶  tasks  ├─▶ code_gen │
│          │ │          │ │          │ │          │ │  flow    │ │ classes  │ │          │ │          │ │   (TDD)  │
└──────────┘ └──────────┘ └────┬─────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────┬─────┘
 feature_     feature_        │        skeleton     data_flow    base_        interfaces   tasks        source
 spec/        build           │        .json        .json        classes      .json        .json        code
 feature_     .json           │        skeleton_    data_flow    .json
 spec.json                    │        summary.txt  _viz.html
                              │
                       ┌──────▼──────┐
                       │ feature_edit│ optional pre-planning edits to feature_tree.json
                       └─────────────┘
                                        ╰───── rpg.json (created → progressively enriched) ─────╯
                                                                            │
                                                                            ▼
                                                                     ┌──────────┐
Surgical edit workflow: Requirements -> RPG update -> Code Update    │ rpg_edit │ optional synchronized RPG + code + dep_graph edits
                                                                     └──▲────▲──┘
                                                                        │    │
Reverse Direction: Code → RPG                                           │    │
                                                                        │    │
┌──────────────────┐         ┌──────────┐       ┌──────────┐            │    │
│ Existing Codebase│────────▶│  encode  │──────▶│update_rpg│────────────┘    │
│                  │         │  (full)  │       │ (manual  │                 │
└──────────────────┘         └────┬─────┘       │ fallback)│                 │
                              rpg.json          └──────────┘                 │
                              dep_graph.json     rpg.json / dep_graph.json   │
                                  │                                          │
                                  └──────────────────────────────────────────┘
                                                  ▲
                                                  │ post-commit hook normally runs incremental updates

MCP Server: search_rpg / explore_rpg / get_node_detail / list_rpg_tree
```

</details>

### CoderMind 실제 사용 예

아래 이미지는 이 저장소에서 생성된 그래프 시각화의 일부입니다. `/cmind.encode` 를 실행한 후 `<workspace>/.cmind/reports/rpg.html` 을 열면 전체 인터랙티브 그래프를 탐색할 수 있습니다. 현재 워크스페이스의 해결된 경로를 보려면 `cmind version` 을 실행하세요.

![CoderMind repository graph visualization](../docs/cmind_visualized_graph.png)

## 설치

### 사전 요구사항

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Git
- 설치 및 인증이 완료된 AI 코딩 에이전트 CLI: [GitHub Copilot](https://docs.github.com/en/copilot) 또는 [Claude Code](https://docs.anthropic.com/en/docs/claude-code/setup)

### CoderMind 설치

```bash
# 영속 설치 (권장)
uv tool install cmind-cli --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind"
cmind check

# 일회성 사용
uvx --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind" cmind init <project-name>
```

`0.1.3` 부터 wheel은 pipeline scripts와 slash-command templates를 packaged assets로 함께 제공하므로, `cmind init` 은 오프라인 환경(air-gapped 환경, 회사 프록시 환경 등)에서도 동작합니다.

## Quick Start: 새 저장소

요구사항을 새 코드베이스로 만들고 싶을 때 이 경로를 사용하세요.

> [!WARNING]
> 생성 코드 양이 많은 프로젝트의 경우, `/cmind.design_interfaces` 와 `/cmind.code_gen` 의 실행 시간이 길어질 수 있습니다. 예시: 100개의 feature는 약 30분이 걸립니다.

1. 새 프로젝트를 초기화합니다:

   ```bash
   cmind init my-project
   cd my-project
   ```

   자주 사용하는 변형:

   ```bash
   cmind init my-project --ai claude --script sh
   cmind init my-project --ai copilot
   ```

2. **[선택]** 요구사항 문서를 `my-project/docs/` 에 둡니다.

3. 프로젝트 디렉터리에서 AI 코딩 에이전트를 실행합니다.

4. 포워드 파이프라인을 실행합니다:

   ```text
   /cmind.feature_construct <feature description>
   [Optional] /cmind.feature_edit <edit instructions>
   /cmind.plan
   /cmind.code_gen
   [Optional] /cmind.rpg_edit <edit instructions>
   ```

> [!IMPORTANT]
> **Coding Agent마다 호출 방식이 조금씩 다릅니다**:
>
> - **Claude Code**: 채팅에 직접 `/cmind.feature_construct ...` 을 입력하면 slash command가 인식되어 해당 workflow가 트리거됩니다.
> - **GitHub Copilot CLI**: slash command는 지원하지 않으나(커스텀 agent는 지원), 먼저 `/agent cmind.feature_construct` 으로 대상 agent로 전환한 다음 `start` 를 입력해 내장된 workflow를 실행합니다.

CoderMind은 `~/.cmind/workspaces/<workspace-id>/data/rpg.json` 을 점진적으로 생성하고, 이를 사용해 요구사항, 계획 산출물, 생성된 코드, 의존성 정보를 정합 상태로 유지합니다. 워크스페이스의 소스 파일은 오염되지 않습니다.

## Quick Start: 기존 저장소

이미 저장소가 있고, AI 에이전트가 RPG 컨텍스트로 이해하거나 편집하기를 원할 때 이 경로를 사용하세요.

> [!WARNING]
> 큰 프로젝트의 경우, `cmind init . --encode` 와 `/cmind.encode` 의 실행 시간이 길어질 수 있습니다. 예시: 200개 소스 파일은 약 100분이 걸립니다.

1. 저장소 루트에서 CoderMind을 초기화하고 초기 그래프를 생성합니다:

   ```bash
   cd existing-repo/
   cmind init . --encode    # --encode 는 현재 코드로부터 RPG를 생성합니다
   ```

   비어 있지 않은 디렉터리에 대한 확인 프롬프트를 건너뛰려면:

   ```bash
   cmind init . --force --encode
   ```

2. 저장소에서 AI 코딩 에이전트를 실행합니다.

3. **[선택]** MCP 도구와 슬래시 커맨드를 통해 생성된 RPG를 사용합니다. 아래 명령은 수동으로 실행할 때만 필요합니다:

   ```text
   /cmind.encode                                  # 필요할 때 전체 RPG 재구축
   /cmind.update_rpg                              # 수동 증분 업데이트 (폴백)
   /cmind.rpg_edit <edit instructions>            # 그래프 인식 코드 편집
   ```

4. 각 commit 후, CoderMind이 설치한 git hook이 `cmind hook <name>` 디스패처를 자동으로 호출해 RPG를 업데이트하고 코드 변경과 정합된 상태로 유지합니다. hook이 실패하거나 건너뛰어진 경우 `/cmind.update_rpg` 를 수동으로 실행하세요.

## `cmind init` 이후 일어나는 일

`cmind init` 은 소스 파일을 수정하지 않습니다. 또한 **워크스페이스에 런타임 상태를 기록하지도 않습니다**. 워크스페이스에는 command 정의, MCP 구성, hooks만 추가합니다. CoderMind의 런타임 데이터(산출물, 로그)는 home-side 디렉터리 `~/.cmind/workspaces/<workspace-id>/` 아래에 배치되며, `<workspace-id>` 는 워크스페이스의 절대 경로에서 파생된 가독성 있는 slug입니다 (예: `home-hys-projects-myrepo`).

```text
my-project/
├── docs/                 # /cmind.feature_construct 용 선택적 요구사항 문서
├── .github/ or .claude/  # Coding Agent 커맨드 정의 및 설정
├── .vscode/              # 해당하는 경우 Copilot/VS Code MCP 구성
├── .cmind/              # 생성된 리포트와 설정 파일
└── .git/hooks/           # cmind init 이 설치하는 post-commit / post-merge (각 hook은 단 한 줄: `cmind hook <name>`)
```

전체 레이아웃과 데이터 파일 참조는 [docs/project-structure.md](docs/project-structure.md) 를 참조하세요.

## CoderMind 업데이트

```bash
uv tool install cmind-cli \
   --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind" \
   --force \
   --reinstall

# 기존 워크스페이스 업데이트
cd <your-workspace>
cmind update
```

## 지원 플랫폼

**Coding Agent 지원**:

| Agent          | CLI 사용 | VS Code 확장 사용 |
| -------------- | -------- | ----------------- |
| Claude Code    | ✅        | ✅                 |
| GitHub Copilot | ✅        | ✅                 |
| Codex          | ⌛        | ⌛                 |

**운영 체제 지원**:

| 운영 체제 | 상태 |
| --------- | ---- |
| Linux     | ✅    |
| macOS     | ⌛    |
| Windows   | ⌛    |

## 문서

- [슬래시 커맨드 레퍼런스](docs/commands.md) — 모든 `/cmind.*` 커맨드의 입력, 출력, 예시.
- [CLI 레퍼런스](docs/cli-reference.md) — `cmind init`, `cmind update`, `cmind check`, `cmind version` 및 모든 옵션.
- [구성](docs/configuration.md) — AI 어시스턴트 설정, MCP 등록, 훅, 자동 승인 및 트러블슈팅.
- [프로젝트 구조](docs/project-structure.md) — CoderMind이 생성하는 파일과 디렉터리.

## 예정된 기능

- **더 간단한 생성 커맨드:** 현재의 다단계 생성 흐름을 `/cmind.generate_repo`, `/cmind.generate_feature` 등 더 적은 커맨드로 통합합니다. `/cmind.plan` 은 0.1.4 에서 출시되었습니다.
- **다국어 지원:** Go, C++, Rust, JavaScript/TypeScript 등을 추가로 지원합니다.
- **더 많은 플랫폼 통합:** 다양한 시스템에서 서로 다른 AI 코딩 에이전트의 CLI 및 VS Code 확장 워크플로에 걸쳐 CoderMind을 지원합니다.

## 트러블슈팅

**AI 어시스턴트 CLI를 찾을 수 없음:** `cmind check` 를 실행하고, 선택한 어시스턴트 CLI를 설치 및 인증한 다음 `cmind init` 또는 `cmind update` 를 다시 실행하세요.

## 라이선스

MIT License — 자세한 내용은 [LICENSE](LICENSE) 참조.

## 감사의 말

[GitHub Spec-Kit](https://github.com/github/spec-kit) 을 기반으로 합니다.
