<h1 align="center">RPG-Kit</h1>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a> |
  <a href="README.ja-JP.md">日本語</a> |
  <a href="README.ko-KR.md">한국어</a> |
  <a href="README.hi-IN.md">हिन्दी</a>
</p>

## コーディングエージェントに、編集する前にプランを立てさせる

コーディングエージェントはローカルな編集には強いものの、リポジトリレベルのタスクは安定した計画構造がないと失敗しがちです。要件はドリフトし、アーキテクチャ上の判断は失われ、複数ファイルにまたがる生成は一貫性を欠き、更新は隠れた依存関係を見落とすことがあります。

RPG-Kit は Claude Code と GitHub Copilot に、リポジトリレベルのコーディングのための**永続的な RPG ワークスペース**を提供します。このワークスペースは、要件・機能・アーキテクチャ・ファイル・コードエンティティ・依存関係をつなぐ **Repository Planning Graph (RPG)** を中心に構成されています。

RPG-Kit を使うと、エージェントはグラフ駆動のワークフローで作業できます:

- **Build（構築）**: 要件を RPG プランに変換し、複数ファイルからなるリポジトリを生成する。
- **Understand（理解）**: 既存のリポジトリを RPG にマッピングし、検索・探索・説明する。
- **Update（更新）**: 影響を受ける RPG ノードを特定し、編集プランを立て、コードとグラフを同時に更新する。

### ワークフローを選ぶ

| 目的 | ワークフロー | ここから始める |
|---|---|---|
| 要件から新しいリポジトリを構築する | Build ワークフロー（requirements → RPG → code） | [`クイックスタート: 新規リポジトリ`](#クイックスタート-新規リポジトリ) |
| 既存のリポジトリを理解する | Understand ワークフロー（repository → RPG → search/explore） | [`クイックスタート: 既存リポジトリ`](#クイックスタート-既存リポジトリ) |
| 既存のリポジトリを更新する | Update ワークフロー（change request → affected RPG nodes → edit plan → code/RPG update） | [`クイックスタート: 既存リポジトリ`](#クイックスタート-既存リポジトリ) |

### 詳細なパイプライン

初めて使う方は、このセクションを飛ばして下のクイックスタートから始められます。

<details>
<summary>コマンドレベルの完全なワークフロー図</summary>

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

### RPG-Kit の実例

下の図は、本リポジトリに対して生成されたグラフ可視化の一部です。`/rpgkit.encode` を実行した後、`<workspace>/.rpgkit/reports/rpg.html` を開くと完全なインタラクティブグラフを閲覧できます。現在のワークスペースの解決済みパスを見るには `rpgkit version` を実行してください。

![RPG-Kit repository graph visualization](../docs/rpgkit_visualized_graph.png)

## インストール

### 前提条件

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Git
- インストール済みで認証済みの AI コーディングエージェント CLI: [GitHub Copilot](https://docs.github.com/en/copilot) または [Claude Code](https://docs.anthropic.com/en/docs/claude-code/setup)

### RPG-Kit のインストール

```bash
# 永続インストール（推奨）
uv tool install rpgkit-cli --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=RPG-Kit"
rpgkit check

# 一度きりの使用
uvx --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=RPG-Kit" rpgkit init <project-name>
```

`0.1.3` 以降、wheel には pipeline scripts と slash-command templates が packaged assets として同梱されるため、`rpgkit init` はオフライン環境（air-gapped 環境や企業プロキシ環境など）でも動作します。

## クイックスタート: 新規リポジトリ

要件から新しいコードベースを生成したい場合は、こちらの手順を使います。

> [!WARNING]
> 生成コード量が多いプロジェクトでは、`/rpgkit.design_interfaces` と `/rpgkit.code_gen` の実行に時間がかかることがあります。例として、100 個の feature でおおよそ 30 分かかります。

1. 新しいプロジェクトを初期化します:

   ```bash
   rpgkit init my-project
   cd my-project
   ```

   よく使うバリエーション:

   ```bash
   rpgkit init my-project --ai claude --script sh
   rpgkit init my-project --ai copilot
   ```

2. **[任意]** 要件ドキュメントを `my-project/docs/` に配置します。

3. プロジェクトディレクトリで AI コーディングエージェントを起動します。

4. フォワードパイプラインを実行します:

   ```text
   /rpgkit.feature_spec <feature description>
   /rpgkit.feature_build
   /rpgkit.feature_refactor
   [Optional] /rpgkit.feature_edit <edit instructions>
   /rpgkit.build_skeleton
   /rpgkit.build_data_flow
   /rpgkit.design_base_classes
   /rpgkit.design_interfaces
   /rpgkit.plan_tasks
   /rpgkit.code_gen
   [Optional] /rpgkit.rpg_edit <edit instructions>
   ```

> [!IMPORTANT]
> **コーディングエージェントごとに呼び出し方が異なります**：
>
> - **Claude Code**：チャットにそのまま `/rpgkit.feature_spec ...` と入力します。slash command が認識され、対応する workflow がトリガーされます。
> - **GitHub Copilot CLI**：slash command はサポートされません（カスタム agent はサポート）。まず `/agent rpgkit.feature_spec` で目的の agent に切り替え、その後 `start` と入力して内蔵の workflow を実行します。

RPG-Kit は `~/.rpgkit/workspaces/<workspace-id>/data/rpg.json` を段階的に作成し、それを使って要件・計画成果物・生成コード・依存情報を整合した状態に保ちます。ワークスペースのソースファイルは汚染されません。

## クイックスタート: 既存リポジトリ

すでにリポジトリがあり、AI エージェントに RPG コンテキストで理解または編集させたい場合は、こちらの手順を使います。

> [!WARNING]
> 大きめのプロジェクトでは、`rpgkit init . --encode` と `/rpgkit.encode` の実行に時間がかかることがあります。例として、200 ファイルでおおよそ 100 分かかります。

1. リポジトリのルートで RPG-Kit を初期化し、初期グラフを構築します:

   ```bash
   cd existing-repo/
   rpgkit init . --encode    # --encode は現在のコードから RPG を生成します
   ```

   空でないディレクトリでの確認プロンプトをスキップしたい場合:

   ```bash
   rpgkit init . --force --encode
   ```

2. リポジトリで AI コーディングエージェントを起動します。

3. **[任意]** 生成された RPG を MCP ツールおよびスラッシュコマンド経由で利用します。以下のコマンドは手動で実行する場合にのみ必要です:

   ```text
   /rpgkit.encode                                  # 必要に応じて完全な RPG を再構築
   /rpgkit.update_rpg                              # 手動の増分更新（フォールバック）
   /rpgkit.rpg_edit <edit instructions>            # グラフ認識型のコード編集
   ```

4. 各 commit の後、RPG-Kit がインストールした git hook が `rpgkit hook <name>` ディスパッチャを自動的に呼び出し、RPG を更新してコード変更と整合した状態に保ちます。hook が失敗したりスキップされたりした場合は、`/rpgkit.update_rpg` を手動で実行してください。

## `rpgkit init` の後に起きること

`rpgkit init` はソースファイルを変更しません。また、**ワークスペースにランタイム状態を書き込みません**。ワークスペースには command 定義、MCP 設定、および hooks のみを追加します。RPG-Kit のランタイムデータ（成果物、ログ）は home-side ディレクトリ `~/.rpgkit/workspaces/<workspace-id>/` 下に配置されます。`<workspace-id>` はワークスペースの絶対パスから導出される可読な slug です（例: `home-hys-projects-myrepo`）。

```text
my-project/
├── docs/                 # /rpgkit.feature_spec 用の任意の要件ドキュメント
├── .github/ or .claude/  # Coding Agent のコマンド定義と設定
├── .vscode/              # 該当する場合の Copilot/VS Code MCP 設定
├── .rpgkit/              # 生成されたレポートと設定ファイル
└── .git/hooks/           # rpgkit init が設置する post-commit / post-merge（各 hook は 1 行のみ: `rpgkit hook <name>`）
```

完全なレイアウトとデータファイルのリファレンスは [docs/project-structure.md](docs/project-structure.md) を参照してください。

## RPG-Kit の更新

```bash
uv tool install rpgkit-cli \
   --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=RPG-Kit" \
   --force \
   --reinstall

# 既存のワークスペースを更新
cd <your-workspace>
rpgkit update
```

## 対応プラットフォーム

**Coding Agent サポート**:

| Agent          | CLI 使用 | VS Code 拡張使用 |
| -------------- | -------- | ---------------- |
| Claude Code    | ✅        | ✅                |
| GitHub Copilot | ✅        | ✅                |
| Codex          | ⌛        | ⌛                |

**オペレーティングシステムサポート**:

| OS      | 状態 |
| ------- | ---- |
| Linux   | ✅    |
| macOS   | ⌛    |
| Windows | ⌛    |

## ドキュメント

- [スラッシュコマンドリファレンス](docs/commands.md) — すべての `/rpgkit.*` コマンドの入力・出力・例。
- [CLI リファレンス](docs/cli-reference.md) — `rpgkit init`、`rpgkit update`、`rpgkit check`、`rpgkit version` とすべてのオプション。
- [設定](docs/configuration.md) — AI アシスタントのセットアップ、MCP 登録、フック、自動承認、およびトラブルシューティング。
- [プロジェクト構造](docs/project-structure.md) — RPG-Kit が作成するファイルとディレクトリ。

## 今後の機能

- **よりシンプルな生成コマンド:** 現在の多段階の生成フローを、`/rpgkit.generate_repo`、`/rpgkit.generate_feature`、`/rpgkit.plan` などのより少ないコマンドにまとめます。
- **多言語サポート:** Go、C++、Rust、JavaScript/TypeScript などのサポートを追加します。
- **より多くのプラットフォーム連携:** さまざまなシステム上の異なる AI コーディングエージェントについて、CLI と VS Code 拡張ワークフローを横断して RPG-Kit をサポートします。

## トラブルシューティング

**AI アシスタント CLI が見つからない:** `rpgkit check` を実行し、選択したアシスタント CLI をインストールおよび認証し、`rpgkit init` または `rpgkit update` を再実行してください。

## ライセンス

MIT License — 詳細は [LICENSE](LICENSE) を参照してください。

## 謝辞

[GitHub Spec-Kit](https://github.com/github/spec-kit) を基にしています。
