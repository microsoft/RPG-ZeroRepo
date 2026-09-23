<h1 align="center">CoderMind</h1>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a> |
  <a href="README.ja-JP.md">日本語</a> |
  <a href="README.ko-KR.md">한국어</a> |
  <a href="README.hi-IN.md">हिन्दी</a>
</p>

## 让编码智能体先规划，再编辑

编码智能体擅长局部编辑，但仓库级任务如果缺少稳定的规划结构往往会失败：需求漂移、架构决策丢失、多文件生成前后不一致、更新可能错过隐藏依赖。

CoderMind 为 Claude Code 和 GitHub Copilot 提供一个面向仓库级编码的**持久化 RPG 工作区**。这个工作区围绕一个 **Repository Planning Graph (RPG)** 构建，把需求、功能、架构、文件、代码实体和依赖关系连接在一起。

借助 CoderMind，智能体可以通过图驱动的工作流来工作：

- **构建（Build）**：把需求转换为 RPG 规划，然后生成一个多文件仓库。
- **理解（Understand）**：把已有仓库映射为 RPG，然后搜索、浏览和解释它。
- **更新（Update）**：定位受影响的 RPG 节点，规划编辑，并同步更新代码和图。

### 选择你的工作流

| 目标                 | 工作流                                                       | 从这里开始                                |
| -------------------- | ------------------------------------------------------------ | ----------------------------------------- |
| 从需求构建一个新仓库 | Build 工作流（requirements → RPG → code）                    | [`快速开始：新仓库`](#快速开始新仓库)     |
| 理解一个已有仓库     | Understand 工作流（repository → RPG → search/explore）       | [`快速开始：已有仓库`](#快速开始已有仓库) |
| 更新一个已有仓库     | Update 工作流（change request → affected RPG nodes → edit plan → code/RPG update） | [`快速开始：已有仓库`](#快速开始已有仓库) |

### 详细流水线

新用户可以跳过这一节，直接从下面的「快速开始」开始。

<details>
<summary>完整的命令级工作流图</summary>

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
│                  │         │  (full)  │       │ (explicit│                 │
└──────────────────┘         └────┬─────┘       │ update)  │                 │
                              rpg.json          └──────────┘                 │
                              dep_graph.json     rpg.json / dep_graph.json   │
                                  │                                          │
                                  └──────────────────────────────────────────┘
                                                  ▲
                                                  │ opt-in post-commit hook: deterministic sync only (no AI update)

MCP Server: search_rpg / explore_rpg / get_node_detail / list_rpg_tree
```

</details>

### CoderMind 实际效果

下图是为本仓库生成的图可视化的一部分。运行 `/cmind.encode` 后，可以打开 `<workspace>/.cmind/reports/rpg.html` 浏览完整的交互式图。运行 `cmind version` 可以看到当前工作区的具体路径。

![CoderMind repository graph visualization](../docs/cmind_visualized_graph.png)

## 安装

### 先决条件

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Git
- 一个已安装并完成身份验证的 Coding Agent CLI：[GitHub Copilot](https://docs.github.com/en/copilot) 或 [Claude Code](https://docs.anthropic.com/en/docs/claude-code/setup)

### 安装 CoderMind

```bash
# 持久化安装（推荐）
uv tool install cmind-cli --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind"
cmind check

# 一次性使用
uvx --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind" cmind init <project-name> --ai claude
```

从 `0.1.3` 开始，wheel 会把 pipeline scripts 和 slash-command templates 作为打包资源一起发布，因此 `cmind init` 的模板配置步骤可以离线完成。可选的初始编码（`--encode`）可能调用 AI 服务并需要网络访问；`cmind update` 中的自升级也可能需要网络访问。

示例使用 `--ai claude`；使用 GitHub Copilot 时替换为 `--ai copilot`。交互式初始化可以在省略 `--ai` 时提示选择；非 TTY 初始化必须显式传入 `--ai`，即使已设置环境覆盖值也不例外。

## 快速开始：新仓库

当你希望 CoderMind 把需求转换为新代码库时，使用此路径。

> [!WARNING]
> 对于生成代码量较大的项目，`/cmind.design_interfaces` 和 `/cmind.code_gen` 可能运行较长时间。典型例子：100 个 feature 大约需要 30 分钟。

1. 初始化一个新项目：

   ```bash
   cmind init my-project --ai claude
   cd my-project
   ```

   常见变体：

   ```bash
   cmind init my-project --ai claude --script sh
   cmind init my-project --ai copilot
   ```

2. **[可选]** 把你的需求文档放在 `my-project/docs/`。

3. 在项目目录里启动你的 AI 编码智能体。

4. 运行正向流水线：

   ```text
   /cmind.feature_construct <feature description>
   [Optional] /cmind.feature_edit <edit instructions>
   /cmind.plan
   /cmind.code_gen
   [Optional] /cmind.rpg_edit <edit instructions>
   ```

> [!IMPORTANT]
> **不同 Coding Agent 的调用方式略有不同**：
>
> - **Claude Code**：直接在对话中输入 `/cmind.feature_construct ...`，slash command 会被识别并触发对应 workflow。
> - **GitHub Copilot CLI**：不支持 slash command（但支持自定义 agent），需要先 `/agent cmind.feature_construct` 切换到目标 agent，然后输入 `start` 让它执行内置的 workflow。

CoderMind 会渐进式地在 home-side 运行时目录（`~/.cmind/workspaces/<workspace-id>/data/rpg.json`）里创建 `rpg.json`，并用它把需求、规划产物、生成的代码和依赖信息保持对齐。主要 RPG 数据存放在仓库外，生成的报告仍保留在工作区内。

## 快速开始：已有仓库

当你已经有一个仓库，希望 AI 智能体在 RPG 上下文中理解或编辑它时，使用此路径。

> [!WARNING]
> 对于较大的项目，`cmind init . --ai claude --encode` 和 `/cmind.encode` 可能运行较长时间。典型例子：200 个源文件大约需要 100 分钟。

1. 在仓库根目录初始化 CoderMind 并构建初始图：

   ```bash
   cd existing-repo/
   cmind init . --ai claude --encode # --encode 会根据当前的代码生成 RPG
   ```

   如果你想跳过非空目录的确认提示：

   ```bash
   cmind init . --ai claude --force --encode
   ```

2. 在仓库里启动你的 AI 编码智能体。

3. 【可选】通过 MCP 工具和 slash 命令使用生成的 RPG，以下命令只在手动运行时需要：

   ```text
   /cmind.encode                                  # 需要时重建完整 RPG
   /cmind.update_rpg                              # 显式请求 AI 驱动的增量更新
   /cmind.rpg_edit <edit instructions>            # 图感知的代码编辑
   ```

4. 使用 `/cmind.update_rpg` 显式请求 AI 驱动的图更新。Git hooks 默认关闭；主动启用后也仅在前台执行确定性同步，不进行 AI 更新或启动后台任务。

## `cmind init` 之后会发生什么

`cmind init` 会配置命令定义、工作区标记/配置、MCP 注册和状态集成，不修改源文件。生成的报告保留在工作区内；主要运行时数据（RPG 产物和日志）存放在 `~/.cmind/workspaces/<workspace-id>/` 下，其中 `<workspace-id>` 是根据工作区绝对路径生成的可读 slug（例如 `home-hys-projects-myrepo`）。

```text
my-project/
├── docs/                 # /cmind.feature_construct 的可选需求文档
├── .github/ or .claude/  # AI 助手的命令定义和设置
├── .vscode/              # 适用时的 Copilot/VS Code MCP 和状态集成
├── .cmind/              # 工作区标记/配置和生成的报告
└── .git/hooks/           # 可选的 post-commit / post-merge，仅执行确定性同步
```

CoderMind 的 Git hooks **默认关闭**。需要确定性同步时，必须在**每次** init/update 调用中传入 `--git-hooks`。省略该选项或使用 `--no-git-hooks` 会移除可识别的 CoderMind 托管块，并保留其他用户/团队 hook 内容。CoderMind 托管的 Git hooks 不启动 AI 或后台任务。Claude 的 `SessionStart` 和 Copilot/VS Code 的 `folderOpen` 仍**仅显示状态**，独立于 Git hooks，不受 `--no-git-hooks` 影响。

完整的目录布局和数据文件参考见 [docs/project-structure.md](docs/project-structure.md)。

### 执行配置

仓库中的 `recommended_provider` 以及有效的旧版 `ai_provider` / 精确匹配内置值的 `ai_cli_cmd` 都只是提示，**不构成运行时授权**。Init 将用户的显式选择保存在仓库外的 `~/.cmind/execution/<workspace-hash>/selection.json`，绑定规范化后的工作区路径，与使用 slug 的 RPG 数据存储分离。详见[本地执行选择](docs/configuration.md#user-local-execution-selection)。

克隆到其他路径、移动工作区或换用户后，必须重新选择。受信任的 CI 可通过 `CMIND_AI_PROVIDER` 为进程指定执行提供方；非 TTY 初始化仍需 `--ai`。`cmind update --ai claude` 会显式更改已保存的选择；不带 `--ai` 的 update 保留原选择（未设置则仍不设置），不会根据仓库提示或检测到的集成自动授权。

不安全的原始命令或无效配置会让 init/update 在配置工作区或迁移 hooks 之前的预检中失败。`--ai`、`--force` 和环境覆盖值都不能绕过校验。请按[迁移清单](docs/configuration.md#updating-an-existing-codermind-project)修正配置。

## 更新 CoderMind

先按你的安装方式安装已包含修复的 CLI，例如：

```bash
uv tool install cmind-cli \
  --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind" \
  --force \
  --reinstall
```

然后迁移**每个已有工作区**：先检查并修正无效配置，再运行下面的显式选择命令（如需 Copilot，将 `claude` 换成 `copilot`）。仅安装 CLI 不会清理工作区中的旧 hooks。

```bash
cd <your-workspace>
cmind update --ai claude --no-upgrade --no-git-hooks

# 日常更新保留本地选择；不会补建缺失的执行授权
cmind update
```

手动检查无法识别的旧 hooks，保留无关的用户/团队 hooks。仅在需要确定性同步时，才在每次 update 中传入 `--git-hooks`。详见[迁移清单](docs/configuration.md#updating-an-existing-codermind-project)。

## 支持的平台

**Coding Agent 支持**：

| Agent          | CLI 使用 | VS Code 扩展使用 |
| -------------- | -------- | ---------------- |
| Claude Code    | ✅        | ✅                |
| GitHub Copilot | ✅        | ✅                |
| Codex          | ⌛        | ⌛                |

**操作系统支持**：

| 操作系统 | 状态 |
| -------- | ---- |
| Linux    | ✅    |
| macOS    | ⌛    |
| Windows  | ⌛    |

Windows 仍为部分支持：模板仅支持 `sh`（不支持 `ps`）。AI 启动仅使用已安装的原生 `.exe` 或受支持的已知 Claude/Copilot npm 入口；从不执行 `.cmd`、`.bat` 或 `.ps1` 包装脚本。这并不保证实际发布版本的兼容性。执行遵循提供方的常规审批，不提供广泛的权限绕过；只读 MCP 预授权是独立设置。详见[可执行文件策略](docs/configuration.md#executable-and-permission-policy)和 [MCP 权限](docs/configuration.md#assistant-permissions-and-scope)。

## 文档

- [Slash 命令参考](docs/commands.md) —— 每一个 `/cmind.*` 命令的输入、输出和示例。
- [CLI 参考](docs/cli-reference.md) —— `cmind init`、`cmind update`、`cmind check`、`cmind version` 以及所有选项。
- [配置](docs/configuration.md) —— 本地提供方选择、MCP 权限、可选 hooks、迁移和故障排查。
- [项目结构](docs/project-structure.md) —— CoderMind 创建的文件和目录。

## 即将推出的功能

- **更简化的生成命令**：把当前多步骤的生成流程合并为更少的命令，例如 `/cmind.generate_repo` 和 `/cmind.generate_feature`。`/cmind.plan` 已在 0.1.4 中发布。
- **多语言支持**：增加对 Go、C++、Rust、JavaScript/TypeScript 等的支持。
- **更多平台集成**：在不同系统上跨 CLI 和 VS Code 扩展工作流支持不同的 AI 编码智能体。

## 故障排查

**找不到 AI 助手 CLI**：运行 `cmind check`，安装并完成所选助手 CLI 的身份验证，然后重新运行 `cmind init` 或 `cmind update`。

## 许可证

MIT License —— 详情见 [LICENSE](LICENSE)。

## 致谢

基于 [GitHub Spec-Kit](https://github.com/github/spec-kit)。
