<h1 align="center">CoderMind</h1>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a> |
  <a href="README.ja-JP.md">日本語</a> |
  <a href="README.ko-KR.md">한국어</a> |
  <a href="README.hi-IN.md">हिन्दी</a>
</p>

## コーディングエージェントに、編集する前にプランを立てさせる

コーディングエージェントはローカルな編集には強いものの、リポジトリレベルのタスクは安定した計画構造がないと失敗しがちです。要件はドリフトし、アーキテクチャ上の判断は失われ、複数ファイルにまたがる生成は一貫性を欠き、更新は隠れた依存関係を見落とすことがあります。

CoderMind は Claude Code と GitHub Copilot に、リポジトリレベルのコーディングのための**永続的な RPG ワークスペース**を提供します。このワークスペースは、要件・機能・アーキテクチャ・ファイル・コードエンティティ・依存関係をつなぐ **Repository Planning Graph (RPG)** を中心に構成されています。

CoderMind を使うと、エージェントはグラフ駆動のワークフローで作業できます:

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

### CoderMind の実例

下の図は、本リポジトリに対して生成されたグラフ可視化の一部です。`/cmind.encode` を実行した後、`<workspace>/.cmind/reports/rpg.html` を開くと完全なインタラクティブグラフを閲覧できます。現在のワークスペースの解決済みパスを見るには `cmind version` を実行してください。

![CoderMind repository graph visualization](../docs/cmind_visualized_graph.png)

## インストール

### 前提条件

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Git
- インストール済みで認証済みの AI コーディングエージェント CLI: [GitHub Copilot](https://docs.github.com/en/copilot) または [Claude Code](https://docs.anthropic.com/en/docs/claude-code/setup)

### CoderMind のインストール

```bash
# 永続インストール（推奨）
uv tool install cmind-cli --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind"
cmind check

# 一度きりの使用
uvx --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind" cmind init <project-name> --ai claude
```

`0.1.3` 以降、wheel には pipeline scripts と slash-command templates が packaged assets として同梱されるため、`cmind init` のテンプレート配置ステップはオフラインで実行できます。任意の初期エンコード（`--encode`）は AI サービスを呼び出し、ネットワークアクセスを必要とする場合があります。`cmind update` による自己アップグレードもネットワークアクセスを必要とする場合があります。

例では `--ai claude` を使います。GitHub Copilot の場合は `--ai copilot` に置き換えてください。対話型の初期化では `--ai` を省略すると選択を求められますが、非 TTY の初期化では環境変数による指定があっても明示的な `--ai` が必要です。

## クイックスタート: 新規リポジトリ

要件から新しいコードベースを生成したい場合は、こちらの手順を使います。

> [!WARNING]
> 生成コード量が多いプロジェクトでは、`/cmind.design_interfaces` と `/cmind.code_gen` の実行に時間がかかることがあります。例として、100 個の feature でおおよそ 30 分かかります。

1. 新しいプロジェクトを初期化します:

   ```bash
   cmind init my-project --ai claude
   cd my-project
   ```

   よく使うバリエーション:

   ```bash
   cmind init my-project --ai claude --script sh
   cmind init my-project --ai copilot
   ```

2. **[任意]** 要件ドキュメントを `my-project/docs/` に配置します。

3. プロジェクトディレクトリで AI コーディングエージェントを起動します。

4. フォワードパイプラインを実行します:

   ```text
   /cmind.feature_construct <feature description>
   [Optional] /cmind.feature_edit <edit instructions>
   /cmind.plan
   /cmind.code_gen
   [Optional] /cmind.rpg_edit <edit instructions>
   ```

> [!IMPORTANT]
> **コーディングエージェントごとに呼び出し方が異なります**：
>
> - **Claude Code**：チャットにそのまま `/cmind.feature_construct ...` と入力します。slash command が認識され、対応する workflow がトリガーされます。
> - **GitHub Copilot CLI**：slash command はサポートされません（カスタム agent はサポート）。まず `/agent cmind.feature_construct` で目的の agent に切り替え、その後 `start` と入力して内蔵の workflow を実行します。

CoderMind は `~/.cmind/workspaces/<workspace-id>/data/rpg.json` を段階的に作成し、それを使って要件・計画成果物・生成コード・依存情報を整合した状態に保ちます。主要な RPG データはリポジトリ外に保存され、生成されたレポートはワークスペース内に残ります。

## クイックスタート: 既存リポジトリ

すでにリポジトリがあり、AI エージェントに RPG コンテキストで理解または編集させたい場合は、こちらの手順を使います。

> [!WARNING]
> 大きめのプロジェクトでは、`cmind init . --ai claude --encode` と `/cmind.encode` の実行に時間がかかることがあります。例として、200 ファイルでおおよそ 100 分かかります。

1. リポジトリのルートで CoderMind を初期化し、初期グラフを構築します:

   ```bash
   cd existing-repo/
   cmind init . --ai claude --encode    # --encode は現在のコードから RPG を生成します
   ```

   空でないディレクトリでの確認プロンプトをスキップしたい場合:

   ```bash
   cmind init . --ai claude --force --encode
   ```

2. リポジトリで AI コーディングエージェントを起動します。

3. **[任意]** 生成された RPG を MCP ツールおよびスラッシュコマンド経由で利用します。以下のコマンドは手動で実行する場合にのみ必要です:

   ```text
   /cmind.encode                                  # 必要に応じて完全な RPG を再構築
   /cmind.update_rpg                              # AI による増分更新を明示的に要求
   /cmind.rpg_edit <edit instructions>            # グラフ認識型のコード編集
   ```

4. AI によるグラフ更新は `/cmind.update_rpg` で明示的に要求してください。Git hooks はデフォルトで無効です。有効にしてもフォアグラウンドで決定論的な同期を行うだけで、AI 更新やバックグラウンド処理は起動しません。

## `cmind init` の後に起きること

`cmind init` はソースファイルを編集せず、コマンド定義、ワークスペースのマーカー/設定、MCP 登録、ステータス連携を設定します。生成されたレポートはワークスペース内に保存され、主要なランタイムデータ（RPG 成果物とログ）は `~/.cmind/workspaces/<workspace-id>/` 下に配置されます。`<workspace-id>` はワークスペースの絶対パスから導出される可読な slug です（例: `home-hys-projects-myrepo`）。

```text
my-project/
├── docs/                 # /cmind.feature_construct 用の任意の要件ドキュメント
├── .github/ or .claude/  # Coding Agent のコマンド定義と設定
├── .vscode/              # 該当する場合の Copilot/VS Code MCP とステータス連携
├── .cmind/              # ワークスペースのマーカー/設定と生成されたレポート
└── .git/hooks/           # 任意の post-commit / post-merge、決定論的な同期のみ
```

CoderMind の Git hooks は**デフォルトで無効**です。決定論的な同期を有効にするには、init/update の**実行ごとに** `--git-hooks` を指定します。省略するか `--no-git-hooks` を指定すると、認識できる CoderMind 管理ブロックだけを削除し、その他のユーザー/チームの hook 内容は保持します。CoderMind が管理する Git hooks は AI やバックグラウンド処理を起動しません。Claude の `SessionStart` と Copilot/VS Code の `folderOpen` は引き続き**ステータス表示のみ**で、Git hooks とは独立しており、`--no-git-hooks` の影響を受けません。

完全なレイアウトとデータファイルのリファレンスは [docs/project-structure.md](docs/project-structure.md) を参照してください。

### 実行設定

リポジトリの `recommended_provider` と、有効な旧形式の `ai_provider` / 組み込み値に完全一致する `ai_cli_cmd` は推奨情報にすぎず、**実行を許可するものではありません**。Init はユーザーの明示的な選択をリポジトリ外の `~/.cmind/execution/<workspace-hash>/selection.json` に保存します。正規化されたワークスペースパスに紐づき、slug ベースの RPG データとは別です。[ローカル実行選択](docs/configuration.md#user-local-execution-selection)を参照してください。

別のパスへのクローン、ワークスペースの移動、別ユーザーでの利用では再選択が必要です。信頼できる CI はプロセスの実行に `CMIND_AI_PROVIDER` を使えますが、非 TTY の初期化には引き続き `--ai` が必要です。`cmind update --ai claude` は保存済みの選択を明示的に変更します。`--ai` なしの update は選択を保持し（未設定なら未設定のまま）、リポジトリの推奨情報や検出した連携から実行を自動許可しません。

安全でない生のコマンド指定や無効な設定があると、init/update はワークスペースの設定や hook 移行前の事前検証で失敗します。`--ai`、`--force`、環境変数による上書きでも検証は回避できません。[移行チェックリスト](docs/configuration.md#updating-an-existing-codermind-project)に従って設定を修正してください。

## CoderMind の更新

まず、利用しているインストール方法で修正を含む CLI をインストールしてください。例:

```bash
uv tool install cmind-cli \
   --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind" \
   --force \
   --reinstall
```

次に、**既存の各ワークスペース**を移行します。無効な設定を確認・修正してから、以下で明示的に選択してください（Copilot を使う場合は `claude` を `copilot` に置き換えます）。CLI のインストールだけでは、ワークスペースの古い hooks は削除されません。

```bash
cd <your-workspace>
cmind update --ai claude --no-upgrade --no-git-hooks

# 通常の更新はローカル選択を保持し、未設定の実行許可は作成しない
cmind update
```

認識されない旧 hooks は手動で確認し、無関係なユーザー/チームの hooks は保持してください。決定論的な同期が必要な場合のみ、update ごとに `--git-hooks` を指定します。[移行チェックリスト](docs/configuration.md#updating-an-existing-codermind-project)を参照してください。

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

Windows 対応は引き続き限定的です。テンプレートは `sh` のみで、`ps` は未対応です。AI の起動にはインストール済みのネイティブ `.exe` または対応する既知の Claude/Copilot npm エントリのみを使い、`.cmd`、`.bat`、`.ps1` ラッパーは実行しません。実際のリリースとの互換性を保証するものではありません。通常のプロバイダー承認が必要で、包括的な権限回避はありません。読み取り専用 MCP の事前承認は別の設定です。[実行ファイルポリシー](docs/configuration.md#executable-and-permission-policy)と [MCP 権限](docs/configuration.md#assistant-permissions-and-scope)を参照してください。

## ドキュメント

- [スラッシュコマンドリファレンス](docs/commands.md) — すべての `/cmind.*` コマンドの入力・出力・例。
- [CLI リファレンス](docs/cli-reference.md) — `cmind init`、`cmind update`、`cmind check`、`cmind version` とすべてのオプション。
- [設定](docs/configuration.md) — ローカルのプロバイダー選択、MCP 権限、任意の hooks、移行、トラブルシューティング。
- [プロジェクト構造](docs/project-structure.md) — CoderMind が作成するファイルとディレクトリ。

## 今後の機能

- **よりシンプルな生成コマンド:** 現在の多段階の生成フローを、`/cmind.generate_repo` や `/cmind.generate_feature` などのより少ないコマンドにまとめます。`/cmind.plan` は 0.1.4 でリリース済みです。
- **多言語サポート:** Go、C++、Rust、JavaScript/TypeScript などのサポートを追加します。
- **より多くのプラットフォーム連携:** さまざまなシステム上の異なる AI コーディングエージェントについて、CLI と VS Code 拡張ワークフローを横断して CoderMind をサポートします。

## トラブルシューティング

**AI アシスタント CLI が見つからない:** `cmind check` を実行し、選択したアシスタント CLI をインストールおよび認証し、`cmind init` または `cmind update` を再実行してください。

## ライセンス

MIT License — 詳細は [LICENSE](LICENSE) を参照してください。

## 謝辞

[GitHub Spec-Kit](https://github.com/github/spec-kit) を基にしています。
