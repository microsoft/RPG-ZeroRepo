# RepoCraft: Benchmark for Repository-Level Code Generation

RepoCraft is a benchmark for evaluating repository-level code generation, derived from the paper *"RPG: A Repository Planning Graph for Unified and Scalable Codebase Generation"*. It consists of **1,052 tasks** across **6 real-world Python projects**, assessing whether AI agents can generate repositories that are functionally complete, algorithmically correct, and at real-world scale.

## Overview

RepoCraft evaluates generated repositories along four dimensions:

| Metric | Description |
|--------|-------------|
| **Functionality Coverage** | Proportion of reference feature categories covered |
| **Functionality Novelty** | Proportion of generated features outside reference taxonomy |
| **Functionality Accuracy** | Pass Rate (tests passed) and Voting Rate (semantic checks passed) |
| **Code-Level Statistics** | File count, Lines of Code (LOC), Token count |

### Reference Repositories

| Original | Anonymized Name | Domain | Tasks |
|----------|----------------|--------|-------|
| scikit-learn | MLKit-Py | Machine Learning | 236 |
| pandas | TableKit | Data Analysis | 175 |
| sympy | SymbolicMath | Symbolic Computation | 192 |
| statsmodels | StatModeler | Statistical Modeling | 234 |
| requests | HttpEasy | HTTP Client | 50 |
| django | PyWebEngine | Web Framework | 165 |

Repository names are paraphrased to prevent pretraining data leakage.

---

## Directory Structure

```
repocraft/
├── __init__.py                   # Top-level exports
├── README.md                     # This file
│
├── benchmark/                    # Benchmark construction pipeline
│   ├── __init__.py
│   ├── __main__.py               # Allows `python -m repocraft.benchmark`
│   ├── main.py                   # Unified CLI: parse → refactor → sample → generate
│   ├── main_parse_test.py        # Standalone CLI for test parsing
│   ├── parse_test.py             # Test tree parsing with LLM
│   ├── refactor_test_tree.py     # Categorize flat test tree by directory structure
│   ├── sample.py                 # Hierarchical test sampling
│   ├── generate_query.py         # Task query generation with LLM
│   └── prompt.py                 # System prompts for parsing
│
├── coverage/                     # Coverage evaluation
│   ├── coverage.py               # Embedding-based coverage metrics
│   ├── sys_prompt.py             # Prompts for coverage evaluation
│   └── gt_repo_tree/             # Ground-truth feature trees (JSON)
│       ├── HttpEasy.json
│       ├── MLKit-Py.json
│       ├── PyWebEngine.json
│       ├── StatModeler.json
│       ├── SymbolicMath.json
│       └── TableKit.json
│
├── framework/                    # Accuracy Evaluation framework
│   ├── __init__.py
│   ├── eval_framework.py         # Main evaluation orchestrator
│   ├── writing_code.py           # Test generation agent
│   ├── sys_prompt.py             # Prompts (localization, voting, coding)
│   ├── utils.py                  # Utility functions
│   └── docker/                   # Docker-based test execution
│       ├── __init__.py
│       ├── repo_docker.py        # Docker container management
│       └── eval_docker.py        # Docker environment for evaluation
│
├── evaluation.py                 # Batch result evaluation & reporting
└── run.py                        # CLI entry point for evaluation run
```

---

## How to Build the RepoCraft Benchmark

The benchmark construction follows a 5-stage pipeline:

```
Reference Repository (e.g., scikit-learn)
    │
    ▼
[Stage 1] Parse Test Tree ──────────── parse_test.py
    │   Collect test files, parse into classes/functions,
    │   use LLM to group test methods by algorithm
    ▼
Flat Feature-Grouped Test Tree (JSON)
    │
    ▼
[Stage 2] Refactor Test Tree ──────── refactor_test_tree.py
    │   Categorize flat file structure into tree
    │   by meaningful directory names
    ▼
Categorized Test Tree (JSON)
    │
    ▼
[Stage 3] Sample Tests ─────────────── sample.py
    │   Hierarchical 3-level sampling:
    │   files → classes → features
    ▼
Sampled Test Subset (JSON)
    │
    ▼
[Stage 4] Generate Task Queries ─────── generate_query.py
    │   For each test group, use LLM to produce:
    │   - Algorithm description
    │   - Natural language task query
    ▼
Task Queries (JSON) ← Ready for evaluation
    │
    ▼
[Stage 5] Evaluate ──────────────────── run.py + framework/
    │   Localization → Voting → Test execution
    ▼
Results
```

You can run the entire benchmark construction pipeline (Stages 1-4) with a single command:

```bash
python -m repocraft.benchmark pipeline \
    --repo_dir /path/to/scikit-learn \
    --output_dir ./all_results \
    --repo_name sklearn \
    --max_parse_workers 4 \
    --max_query_workers 6
```

Or run individual stages separately (see below).

### Stage 1: Parse Test Tree

Parse the reference repository's test suite into a hierarchical feature tree. This identifies what algorithms/functions each test validates.

```bash
python -m repocraft.benchmark.main_parse_test parse \
    --repo_dir /path/to/scikit-learn \
    --result_path ./all_results/result_tests/sklearn.json \
    --max_workers 4 \
    --max_iterations 10
```

**What it does:**
1. Scans the repository for test files (files in `tests/` directories)
2. Parses each test file into code units (classes, functions) using AST
3. Uses an LLM to group test methods by the core algorithm/functionality they test
4. Outputs a hierarchical JSON structure

**Output format:**
```json
{
  "tests/test_svm.py": {
    "class TestSVC": {
      "svc_prediction": ["test_svc_predict", "test_svc_predict_proba"],
      "svc_parameters": ["test_svc_kernel", "test_svc_gamma"]
    },
    "functions": {
      "svr_basic": ["test_svr_predict"]
    }
  }
}
```

**How the LLM grouping works:**
- The LLM receives the full test class code and is instructed to group methods by the core algorithm they validate
- Grouping is semantic-first: tests checking different aspects of the same algorithm are merged
- Each test method appears exactly once
- Names use `snake_case` and refer to the public API or canonical algorithm name

### Stage 2: Refactor Test Tree

Convert the flat file-based test tree (keyed by absolute paths) into a categorized tree organized by meaningful directory names (e.g., `metrics`, `clustering`, `preprocessing`).

```bash
python -m repocraft.benchmark.main refactor \
    --parsed_test ./all_results/result_tests/sklearn.json \
    --result_path ./all_results/refactored_test/sklearn.json
```

**What it does:**
1. Reads the flat parsed test tree from Stage 1
2. For each file path, extracts the most meaningful directory name (skipping `test`/`tests` directories)
3. Groups test files under their functional category
4. Outputs a JSON with both the original (`files`) and refactored (`refactor`) structures

**Output format:**
```json
{
  "files": { ... },
  "refactor": {
    "svm": {
      "test_svm": { "class TestSVC": { ... }, "functions": { ... } }
    },
    "metrics": {
      "test_pairwise": { ... },
      "test_regression": { ... }
    }
  }
}
```

### Stage 3: Sample Tests

Apply hierarchical sampling to select a representative subset from the refactored test tree.

```bash
python -m repocraft.benchmark.main sample \
    --refactored_test ./all_results/refactored_test/sklearn.json \
    --result_path ./all_results/sampled_test/sample_sklearn.json \
    --num_files 12 \
    --num_classes_per_file 20 \
    --num_modules_per_class 10
```

**Sampling strategy:**
- **Level 1 (Files):** Randomly select `num_files` test files, excluding base/issue files
- **Level 2 (Classes/Functions):** From each file, randomly select `num_classes_or_functions_per_file` test classes or function groups
- **Level 3 (Features):** From each class, randomly select `num_modules_per_class` algorithm features

This ensures balanced coverage across the repository's functional categories.

### Stage 4: Generate Task Queries

For each sampled test group, generate a natural language task description and algorithm description using an LLM.

```bash
python -m repocraft.benchmark.main generate \
    --sampled_test ./all_results/sampled_test/sample_sklearn.json \
    --parsed_test ./all_results/result_tests/sklearn.json \
    --result_path ./all_results/task_results/sklearn.json \
    --max_workers 6
```

**What it does:**
1. Reads the sampled test JSON and the full parsed test JSON
2. For each sampled test group, extracts the actual test code
3. Uses an LLM to generate:
   - **Algorithm Description** (`alg_description`): A high-level, abstract description of what the algorithm does, without implementation details
   - **Task Query** (`task_query`): A natural language query like "You are testing an algorithm that..."

**Output format (one task):**
```json
{
    "category": "metrics",
    "file": "tests/test_metrics.py",
    "module": "class TestRegressionMetrics",
    "cap": "mean_squared_error",
    "functions": ["test_mse_basic", "test_mse_multioutput"],
    "query_code": "def test_mse_basic():\n    ...",
    "alg_description": "Computing the mean squared error between predicted and actual values, supporting both single-output and multi-output regression scenarios.",
    "task_query": "You are testing an algorithm that calculates the mean squared error (MSE) between predicted values and ground truth, with support for sample weights and multi-output averaging strategies.",
    "id": "sklearn-0042"
}
```

### Stage 5: Evaluate Generated Repositories

Run the evaluation pipeline on a generated repository against the task set.

```bash
python -m repocraft.run \
    --tasks_file ./all_results/task_results/sklearn.json \
    --method_path /path/to/generated/MLKit-Py \
    --cache_dir ./eval_cache \
    --mnt_dir /tmp/workspace \
    --model_loc_vote o3-mini \
    --model_test o3-mini \
    --max_loc_iters 40 \
    --max_coding_iters 15 \
    --max_retries 5 \
    --image_name zerorepo \
    --skip_existing \
    --verbose
```

**The evaluation pipeline has 3 stages:**

#### Stage 5a: Localization
An RPGAgent navigates the generated repository to find functions/classes that implement the target algorithm. It uses:
- RPG-guided search (functionality-based fuzzy matching)
- Repository code view (inspect function bodies)
- Dependency exploration (trace edges for related modules)

#### Stage 5b: Majority-Vote Validation
5 rounds of LLM-based voting to verify whether the localized code actually implements the target algorithm. If voting fails, the pipeline retries localization (up to 3 times).

#### Stage 5c: Test Adaptation and Execution
The ground-truth test is adapted to match the naming/structure of the localized code, then executed in a Docker container. The test result determines functional correctness.

### Evaluate Results

After running evaluation, analyze the results:

```bash
python -m repocraft.evaluation \
    --base-dir ./exp_results \
    --models gpt-4.1 gpt-5-mini \
    --exp-types docs ref \
    --show-failed \
    --output results.json
```

This produces summary tables with pass rates, voting rates, and per-repository breakdowns.

---

## Coverage Evaluation

To evaluate how well a generated repository covers the reference feature taxonomy:

```python
from repocraft.coverage.coverage import SubtreeCoverageEvaluator

evaluator = SubtreeCoverageEvaluator(
    model_id="Alibaba-NLP/gte-Qwen2-7B-instruct",
    outlier_tag="new_features"
)

# predicted_trees: feature tree from generated repository
# gt_tree: ground-truth feature tree from coverage/gt_repo_tree/
results = evaluator.evaluate(
    predicted_trees=predicted_trees,
    gt_tree=gt_tree
)
```

Ground-truth feature trees for all 6 repositories are provided in `coverage/gt_repo_tree/`.

---

## End-to-End Example: Building the Benchmark for scikit-learn

### Option A: Full Pipeline (single command)

```bash
# Stages 1-4 in one command
python -m repocraft.benchmark pipeline \
    --repo_dir /path/to/scikit-learn \
    --output_dir ./all_results \
    --repo_name sklearn \
    --max_parse_workers 4 \
    --max_query_workers 6

# Stage 5: Run evaluation on generated repository
python -m repocraft.run \
    --tasks_file ./all_results/task_results/sklearn.json \
    --method_path /path/to/generated/MLKit-Py \
    --cache_dir ./eval_cache \
    --model o3-mini \
    --skip_existing

# Analyze results
python -m repocraft.evaluation \
    --base-dir ./eval_cache \
    --show-failed
```

### Option B: Step by Step

```bash
# Stage 1: Parse test tree
python -m repocraft.benchmark.main parse \
    --repo_dir /path/to/scikit-learn \
    --result_path ./all_results/result_tests/sklearn.json \
    --max_workers 4

# Stage 2: Refactor into categorized tree
python -m repocraft.benchmark.main refactor \
    --parsed_test ./all_results/result_tests/sklearn.json \
    --result_path ./all_results/refactored_test/sklearn.json

# Stage 3: Sample tests
python -m repocraft.benchmark.main sample \
    --refactored_test ./all_results/refactored_test/sklearn.json \
    --result_path ./all_results/sampled_test/sample_sklearn.json

# Stage 4: Generate task queries
python -m repocraft.benchmark.main generate \
    --sampled_test ./all_results/sampled_test/sample_sklearn.json \
    --parsed_test ./all_results/result_tests/sklearn.json \
    --result_path ./all_results/task_results/sklearn.json \
    --max_workers 6

# Stage 5: Run evaluation on generated repository
python -m repocraft.run \
    --tasks_file ./all_results/task_results/sklearn.json \
    --method_path /path/to/generated/MLKit-Py \
    --cache_dir ./eval_cache \
    --model o3-mini \
    --skip_existing

# Analyze results
python -m repocraft.evaluation \
    --base-dir ./eval_cache \
    --show-failed
```

---

## Configuration

### LLM Configuration

The pipeline uses LLM for test parsing, query generation, localization, voting, and test adaptation. Configure via:

```python
from zerorepo.rpg_gen.base.llm_client import LLMConfig

# Use a specific model
cfg = LLMConfig(model="o3-mini")

# Or load from JSON/YAML file
cfg = LLMConfig.from_source("path/to/config.json")
```

### Docker Configuration

The evaluation framework runs tests inside Docker containers for isolation:
- **Image**: `zerorepo` (default)
- **Mount**: Test workspace mounted at `/workspace`, repo at `/repo`
- **Conda**: Activates `zerorepo` conda environment inside container

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_loc_iters` | 40 | Max localization iterations per attempt |
| `max_coding_iters` | 15 | Max test generation iterations |
| `max_retries` | 5 | Max localization retry attempts |
| `voting_times` | 5 | Number of voting rounds |
| `context_window` | 10-20 | LLM memory context window |

---

## Dependencies

- `zerorepo`: Repository analysis, LLM client, code parsing
- `docker`: Container management for test execution
- `tiktoken`: Token counting for output truncation
- `networkx`: Graph operations (dependency graphs)
- `torch`, `transformers`: Embedding models for coverage evaluation
- `scikit-learn`: Clustering for coverage matching
- `tqdm`: Progress bars
