# RPG-Encoder

[![arXiv:2602.02084](https://img.shields.io/badge/Paper-arXiv%3A2602.02084-b31a1b)](https://arxiv.org/abs/2602.02084)

RPG-Encoder generalizes the Repository Planning Graph (RPG) from a static generative blueprint into a **unified, high-fidelity representation** for existing repositories. It closes the reasoning loop between comprehension and generation -- generation expands intent into implementation, while comprehension compresses implementation back into intent.

## 1. RPG Encoding: Extracting RPG from Codebases

**Module:** `rpg_parsing/`
**Entry:** `RPGParser.parse_rpg_from_repo()`

Transforms a raw codebase into an actionable RPG through three phases:

### Phase 1: Semantic Lifting

Bridges the granularity mismatch between verbose implementation and functional intent. For each file, the system extracts semantic features for individual functions and classes, mapping them to behavioral signatures while retaining code-level attributes as metadata.

```
Source Code (functions, classes)
    │
    ▼  LLM-based feature extraction (ParseFeatures)
    │
Leaf Nodes: v = (f, m)
    f = semantic feature (e.g., "monotonic trend detector")
    m = metadata (type: function, path: sklearn/isotonic.py, name: check_increasing)
```

- **File-level**: Features of individual units are synthesized into a holistic summary
- **Functional edges** (Efeature) link file nodes to constituent function nodes
- Result: a semantically grounded implementation index

### Phase 2: Semantic Structure Reorganization

Physical folder-file organization is often dictated by technical constraints rather than functional boundaries. This phase recovers the latent functional topology:

1. **Functional Abstraction**: The LLM consumes concise semantic features (not raw code) to induce abstract functional centroids (e.g., "Data Preprocessing", "Model Training")
2. **Hierarchical Aggregation**: Recursively links leaf nodes to centroids via semantic compatibility checks, creating intermediate nodes to bridge granularity gaps

```
Before (physical layout):           After (functional hierarchy):
sklearn/                            Preprocessing Algorithms
├── preprocessing/                  ├── Normalization
│   ├── _data.py                    │   ├── StandardScaler
│   └── _encoders.py                │   └── MinMaxScaler
├── isotonic.py                     ├── Encoding
└── tree/                           │   └── OneHotEncoder
    └── _classes.py                 └── Monotonic Fitting
                                        └── check_increasing
```

### Phase 3: Artifact Grounding

Anchors the abstract hierarchy to physical artifacts:

1. **Metadata Propagation**: Populates directory-scope metadata for high-level nodes via Lowest Common Ancestor (LCA) computation
2. **Dependency Injection**: Injects dependency edges (Edep) via AST analysis (imports, calls), completing the dual-view graph

### Usage

```python
from zerorepo.rpg_encoder import RPGEncoder

# Initial encoding
encoder = RPGEncoder(
    repo_dir="/path/to/repo",
    repo_name="myrepo",
    repo_info="A machine learning library for ...",  # optional
)

rpg, feature_tree, skeleton = encoder.encode(
    max_parse_iters=20,
    max_parse_workers=4,
    refactor_max_iters=20,
    update_dep_graph=True,
)

encoder.save("output/rpg_encoder.json")
```

Or via CLI:

```bash
python parse_rpg.py parse \
    --repo-dir /path/to/repo \
    --repo-name myrepo \
    --save-dir ./output \
    --max-parse-iters 20 \
    --max-parse-workers 4
```

### Output Files

| File | Description |
|------|-------------|
| `rpg_encoder.json` | Full encoder state (loadable for incremental updates) |
| `global_repo_rpg.json` | RPG graph (nodes, edges, dependency graph) |
| `repo_data.json` | Repository metadata, feature tree, and components |
| `skeleton.json` | Repository file/directory skeleton |

---

## 2. RPG Evolution: Incremental Maintenance

**Module:** `rpg_parsing/rpg_evolution.py`
**Entry:** `RPGEvolution.process_diff()`

Maintains the RPG incrementally by parsing commit diffs, avoiding costly full re-generation. This reduces maintenance overhead by **95.7%** compared to re-encoding from scratch.

### Update Protocol

Given a diff between two repository versions, three atomic update operations are applied:

| Diff Type | Operation | Detail |
|-----------|-----------|--------|
| **Deletion** | Remove nodes | Delete nodes for removed files/functions; recursively prune empty parent categories |
| **Modification** | Re-generate features | Update semantic description `f`; relocate node only if LLM detects a functional intent shift |
| **Addition** | Insert new nodes | Create nodes for new entities; match semantics against existing centroids for placement |

After structural updates, dependency edges (Edep) are refreshed via localized AST re-parsing.

### Usage

```python
# Load existing RPG
encoder = RPGEncoder.from_saved(
    save_path="output/rpg_encoder.json",
    cur_repo_dir="/path/to/updated/repo",
)

# Incremental update
rpg = encoder.update(
    last_repo_dir="/path/to/previous/repo",
    update_dep_graph=True,
)

encoder.save("output/rpg_encoder_updated.json")
```

Or via CLI:

```bash
python parse_rpg.py update \
    --repo-dir /path/to/updated/repo \
    --last-repo-dir /path/to/previous/repo \
    --load-path ./output/rpg_encoder.json \
    --save-dir ./output
```

### Key Design Decisions

- **Semantic threshold for relocation**: Minor implementation changes (bug fixes, refactors) do not trigger structural migration. Only when the LLM detects a fundamental change in functional intent (e.g., a utility function evolving into a core algorithm) is the node relocated.
- **Localized dependency refresh**: Only affected ASTs are re-parsed, not the entire repository.

---

## 3. RPG Operation: Unified Reasoning Substrate (RPG Agent)

**Module:** `rpg_agent/`
**Entry:** `RPGAgent.run()`

Deploys the RPG as a unified interface for structure-aware reasoning. The RPG functions as a heterogeneous graph where Functional View (Efeature) and Dependency View (Edep) are partitioned by edge type but share a unified node set, enabling seamless context switching during retrieval.

### Agent Tools

The agent operates with three core tools that leverage the RPG's dual-view structure:

| Tool | Purpose | How It Works |
|------|---------|--------------|
| **SearchNode** | Global node-level retrieval | Matches intent against semantic features `f` or filters metadata `m`. Supports both code search (file paths, qualified names, text keywords) and feature search (functional descriptions). Uses BM25 + substring matching. |
| **FetchNode** | Node-level data retrieval | Given a node, extracts the full attribute tuple `(f, m)` and raw source code for ground-truth inspection. |
| **ExploreRPG** | Cross-view graph traversal | Traverses along edges `E` (upstream/downstream). Dependency edges from AST analysis combined with the semantic hierarchy provide a robust topological skeleton for navigation. |

### Usage

```python
from zerorepo.rpg_encoder.rpg_agent import RPGAgent

agent = RPGAgent(
    llm_cfg=llm_config,
    instance_id="task_001",
    task="The _ovr_decision_function in SVM was not correctly normalizing the sum of votes.",
    repo_dir="/path/to/repo",
    repo_name="sklearn",
    dep_graph=rpg.dep_graph.G,   # networkx MultiDiGraph
    repo_rpg=rpg,                # RPG instance
    max_steps=40,
    context_window=30,
)

result = agent.run()
# result keys: final_results, is_terminate, is_suc,
#              all_traj, action_history, feedback_history,
#              step_token_usage, total_prompt_tokens, total_completion_tokens
```

## 4. Repository Reconstruction (Rebuild)

**Module:** `rebuild.py`
**Entry:** `Rebuild.run()`

Reconstructs a repository from its RPG representation, validating that the RPG preserves sufficient information for faithful reproduction. This serves as a **representational fidelity** test.

### Rebuild Modes

| Mode | Preserves | Redesigns | Use Case |
|------|-----------|-----------|----------|
| `FEATURE_ONLY` | Feature graph | Files, functions | Test if features alone suffice |
| `FEATURE_FILE` | Features + file layout | Function signatures | Test with file-level info |
| `FULL_PRESERVE` | All info | Data flow + file ordering | Guided reconstruction |

### Usage

```python
from zerorepo.rpg_encoder.rebuild import Rebuild, RebuildConfig, RebuildMode

config = RebuildConfig(
    mode=RebuildMode.FULL_PRESERVE,
    llm_config=llm_config,
    skeleton_cfg_path="configs/file_design_config.yaml",
    graph_cfg_path="configs/func_design_config.yaml",
)

rebuilder = Rebuild(
    repo_dir="/path/to/repo",
    repo_name="sklearn",
    checkpoint_dir="./checkpoints",
    config=config,
)

rebuilder.run()
```

### Results on RepoCraft

| Method | Coverage | Pass Rate |
|--------|----------|-----------|
| Documentation baseline | 74.2% | 52.7% |
| **RPG-Encoder (GPT-5-mini)** | **98.5%** | **86.0%** |

RPG-Encoder achieves 98.5% reconstruction coverage, confirming RPG's capacity as a high-fidelity representational substrate.

---

## RPG Data Structure

The RPG is a hierarchical, dual-view graph `G = (V, E)`:

### Nodes

```
V = V_H ∪ V_L

V_H (High-level Nodes):
    - Represent functional areas / abstract categories
    - Have semantic feature f, but no direct code metadata
    - Examples: "Preprocessing Algorithms", "Model Evaluation"

V_L (Low-level Nodes):
    - Represent concrete code entities (files, classes, functions)
    - Each node v = (f, m):
        f = semantic feature ("monotonic trend detector")
        m = metadata (type, file_path, function_name, line_range)
```

### Edges

```
E = E_feature ∪ E_dep

E_feature (Functional Edges):
    - Establish teleological hierarchy (parent → child)
    - "Preprocessing" → "Normalization" → "StandardScaler"

E_dep (Dependency Edges):
    - Map logical interactions (imports, calls)
    - Injected via AST analysis
    - "fit_transform()" calls→ "check_input()"
```

### Example Node

```json
{
  "id": "node_0042",
  "name": "monotonic trend detector",
  "feature_path": "preprocessing_algorithms/monotonic_fitting/monotonic_trend_detector",
  "meta": {
    "type": "function",
    "path": "sklearn/isotonic.py",
    "func_name": "check_increasing",
    "line_range": [15, 48]
  }
}
```

---

## Configuration

### LLM Configuration

All components accept an `LLMConfig` object or a config file path:

```python
from zerorepo.rpg_gen.base import LLMConfig

# Direct configuration
cfg = LLMConfig(model="gpt-4o", provider="openai", api_key="...")

# From file (YAML or JSON)
cfg = LLMConfig.from_source("configs/llm_config.yaml")
```

### Key Parameters

| Parameter | Default | Component | Description |
|-----------|---------|-----------|-------------|
| `max_parse_iters` | 10-20 | Encoding | Max LLM iterations per parsing unit |
| `max_parse_workers` | 4-8 | Encoding | Parallel workers for feature extraction |
| `refactor_max_iters` | 10-20 | Encoding | Max iterations for tree refactoring |
| `min_batch_tokens` | 10,000 | Encoding | Min tokens per parsing batch |
| `max_batch_tokens` | 50,000 | Encoding | Max tokens per parsing batch |
| `max_steps` | 30-40 | Agent | Max agent reasoning steps |
| `context_window` | 30 | Agent | LLM memory context window |
