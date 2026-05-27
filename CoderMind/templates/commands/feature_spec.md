---
description: Create structured feature specifications from user input or documentation files
name: rpgkit.feature_spec
---

## User Input

```text
$ARGUMENTS
```

Text provided after `/rpgkit.feature_spec` will be used as the feature description. If empty, the agent will automatically detect and use files in the `docs/` directory.

## Capabilities

- **Dual Input Modes**: Accepts either directly provided user descriptions or auto-detects `docs/*.md` files
- **Evidence-Based Extraction**: All feature specifications are traceable to source text with line numbers
- **Sub-Document Architecture**: Generates modular Markdown files for improved maintainability
- **Hierarchical Feature Tree**: Supports unlimited nesting levels using numeric outline format
- **Unified Processing Flow**: Both input modes converge to the same downstream workflow

## Output Directory Structure

```text
.rpgkit/data/feature_spec/
├── evidence/                      # Step 2 output
│   ├── user_input.md              # (from user input) or
│   ├── 01_project_charter.md      # (from docs/)
│   ├── 02_requirements_specification.md
│   └── ...
├── feature_spec.md                # Step 3 output (Meta + Background + NFR)
└── features/                      # Step 4 output (Feature Tree)
    ├── FT-001.md
    ├── FT-002.md
    └── ...
```

## Workflow

**Working Directory**: All relative paths are based on the project root.

### Step 1: Determine Input Source

#### 1.1 Check User Input

If `$ARGUMENTS` is **not empty**:

```markdown
## ✓ User Input Detected

- **Mode**: User-provided description
- **Input Length**: <character count> characters
- **Output Directory**: .rpgkit/data/feature_spec/

Processing user-provided feature description.
```

→ Proceed to **Step 2A: Process User Input**

#### 1.2 Check docs/ Directory

If `$ARGUMENTS` is **empty**, check for documentation files:

1. List all `.md` files in the `./docs` directory.

2. If files are found:

   ```markdown
   ## Documentation Detected
   
   No feature description provided. Found the following Markdown files in `docs/` directory:
   
   - 01_project_charter.md
   - 02_requirements_specification.md
   - ...
   
   **Use these documents to create feature specification? (Y/N)**
   ```

   **Wait for user response:**

   - **If "Y"**: → Proceed to **Step 2B: Process Documentation Files**
   - **If "N"**: → Display message and terminate

#### 1.3 No Available Input

If neither user input nor documentation files are available:

```markdown
## No Available Input Source

The feature specification process requires one of the following:

1. **Provide a feature description as input:**

   `/rpgkit.feature_spec <your feature description>`

2. **Place documentation files in the `docs/` directory:**

   Add requirement or design documents to the `docs/` directory, then run:

   `/rpgkit.feature_spec`
```

→ **Terminate agent execution**

---

### Step 2A: Process User Input → Evidence

Convert user input to Evidence file format.

#### 2A.1: Parse User Input

1. **Analyze user input** to identify:
   - Background information (context, goals, scope, design philosophy, etc.)
   - Functional requirements (features, behaviors, operations, interface contracts, etc.)
   - Non-functional requirements (constraints, performance, security, assumptions, risks, etc.)

2. **Assign sequential IDs** in format `user_input-{category}-{sequence}`:
   - Background: `user_input-BG-001`, `user_input-BG-002`, ...
   - FR: `user_input-FR-001`, `user_input-FR-002`, ...
   - NFR: `user_input-NFR-001`, `user_input-NFR-002`, ...

3. **Virtual line numbers**: User input is not a file, use virtual line numbers (split user input text by lines, counting from line 1)

#### 2A.2: Generate Evidence File

Create `.rpgkit/data/feature_spec/evidence/user_input.md`:

```markdown
# Evidence: user_input.md

## Background / Overview

### [user_input-BG-001] L1-L{end}
**Parent Title**: -

> {Background content extracted from user input}

## FR (Functional Requirement)

### [user_input-FR-001] L{start}-L{end}
**Parent Title**: -

> {Functional requirement extracted from user input}

### [user_input-FR-002] L{start}-L{end}
**Parent Title**: -

> {Another functional requirement}

## NFR (Non-Functional Requirement)

### [user_input-NFR-001] L{start}-L{end}
**Parent Title**: -

> {Non-functional requirement extracted from user input}
```

#### 2A.3: Display Progress

```markdown
## ✓ User Input Processing Complete

- **Evidence File**: .rpgkit/data/feature_spec/evidence/user_input.md
- **Background Entries**: <count>
- **FR Entries**: <count>
- **NFR Entries**: <count>
- **Total Evidence**: <total>
```

→ Proceed to **Step 3: Generate Main File**

---

### Step 2B: Process Documentation Files → Evidence

Extract Evidence from each documentation file.

> **Important: Process One Document at a Time**
>
> Process documents **one by one**, completing each before moving to the next.
>
> - Avoid processing all documents at once to prevent context overflow and quality degradation
> - Display progress after completing each document, then continue to the next

#### 2B.1: Process Each Document

For each `.md` file in `docs/`, create a corresponding evidence file.

**ID Format:**

```text
{file_prefix}-{category}-{sequence}
     │           │         │
     │           │         └── 001, 002, 003... (numbered independently per category)
     │           └────── BG / FR / NFR
     └──────────── File name abbreviation or full name
```

**File Prefix Generation Rules:**

1. Remove numeric prefix and extension (e.g., `01_project_charter.md` → `project_charter`)
2. Choose scheme based on length:
   - **≤ 20 characters**: Use full name as prefix
   - **> 20 characters**: Use abbreviation (capitalize first letter of each word)
3. **Handle duplicate prefixes**: If generated prefix matches an existing prefix, append sequence number `_2`, `_3`...

**Examples:**

| Filename | File Prefix | BG ID | FR ID | NFR ID |
| -------- | ----------- | ----- | ----- | ------ |
| `01_project_charter.md` | `project_charter` | `project_charter-BG-001` | `project_charter-FR-001` | `project_charter-NFR-001` |
| `02_requirements_specification.md` | `RS` | `RS-BG-001` | `RS-FR-001` | `RS-NFR-001` |
| `05_interface_data_contract.md` | `IDC` | `IDC-BG-001` | `IDC-FR-001` | `IDC-NFR-001` |

**Abbreviation Generation Rules:**

1. Split by `_` into words
2. Take first letter of each word
3. Convert to uppercase

**Duplicate Prefix Handling Examples:**

| Filename | Generated Prefix | Conflict Handling | Final Prefix |
| -------- | ---------------- | ----------------- | ------------ |
| `docs/api_reference.md` | `api_reference` | No conflict | `api_reference` |
| `docs/sub/api_reference.md` | `api_reference` | Already exists | `api_reference_2` |
| `docs/other/api_reference.md` | `api_reference` | Already exists | `api_reference_3` |

**Special Cases:**

- User input always uses prefix `user_input`

#### 2B.2: Generate File ID Mapping Table

Before processing documents, scan all documents and generate the ID prefix mapping table:

```markdown
## File ID Mapping Table

| # | File Path | File Prefix |
|---|-----------|-------------|
| 1 | docs/01_project_charter.md | `project_charter` |
| 2 | docs/02_requirements_specification.md | `RS` |
| 3 | docs/sub/api_reference.md | `api_reference` |
| 4 | docs/other/api_reference.md | `api_reference_2` |
| ... | ... | ... |

**Total Documents**: <N>
```

This mapping table will be used to trace Evidence sources later.

#### 2B.3: Extract Evidence from Each Document

For each document, read the complete content and extract evidence:

1. **Identify section headers** (e.g., "3.1 DAG Authoring and Definition") - as Parent Title
2. **Record line numbers** for traceability
3. **Preserve original text** verbatim copy
4. **Categorize**: Background / Overview, FR, NFR

**Field Source Description:**

- **Parent Title**: Section header from source (for tracing source location)

**Evidence Category Quick Reference:**

| Category | Definition | Typical Content |
| -------- | ---------- | --------------- |
| **BG** | System context, goals, design philosophy | Project background, system boundaries, design principles, terminology definitions |
| **FR** | What the system **does** | Module responsibilities, APIs, user interactions, data flows, interface contracts |
| **NFR** | System's **quality attributes and constraints** | Performance, security, scalability, assumptions, constraints, technical risks |

**FR Extraction Granularity Requirements:**

- Extract each independent feature point as one Evidence
- Describe "what" not "how"
- Extract each table row feature independently

**Content NOT to Extract:**
Project risks, external risks, operational processes, inter-document references, open questions (undecided items)

#### 2B.4: **Generate Evidence Files**

For each document, create `.rpgkit/data/feature_spec/evidence/{document_name}.md`:

```markdown
# Evidence: {document_name}.md

## Background / Overview

### [{prefix}-BG-001] L{start}-L{end}
**Parent Title**: -

> {Original excerpt}

### [{prefix}-BG-002] L{start}-L{end}
**Parent Title**: {Section header}

> {Original excerpt}

## FR (Functional Requirement)

### [{prefix}-FR-001] L{start}-L{end}
**Parent Title**: {Section header, e.g., 3.1 DAG Authoring & Definition}

> {Original excerpt}

### [{prefix}-FR-002] L{start}-L{end}
**Parent Title**: {Section header}

> {Original excerpt}

## NFR (Non-Functional Requirement)

### [{prefix}-NFR-001] L{start}-L{end}
**Parent Title**: {Section header, e.g., 4.1 Performance}

> {Original excerpt}
```

#### 2B.5: Display Progress

After processing each document:

```markdown
## Document Processing Progress

### ✓ <filename1>.md
- **Evidence File**: .rpgkit/data/feature_spec/evidence/<filename1>.md
- **Background**: <count> | **FR**: <count> | **NFR**: <count>

### ✓ <filename2>.md
- **Evidence File**: .rpgkit/data/feature_spec/evidence/<filename2>.md
- **Background**: <count> | **FR**: <count> | **NFR**: <count>

...

---

**Total Documents**: <N>
**Total Evidence Entries**: <M>
```

→ Proceed to **Step 3: Generate Main File**

---

### Step 3: Generate Main File

Generate the main feature specification file containing Meta, Background, and NFR.

#### 3.1: Read All Evidence Files

Load all `.md` files from `.rpgkit/data/feature_spec/evidence/`.

#### 3.2: Determine Repository Information

Derive from evidence:

- **Repository Name**: Concise name (1-3 words, kebab-case)
- **Repository Purpose**: 1-2 sentences describing the core objective
- **Project Types**: Identify which user-facing surfaces the project exposes.
  Output a list of UPPERCASE tokens drawn from this 8-item whitelist.
  At least one token is required; multiple are allowed when the project
  exposes more than one surface.

  | Token | Use it for |
  | --- | --- |
  | `WEB` | HTTP endpoints that render HTML pages for browsers |
  | `API` | JSON / GraphQL endpoints, no HTML rendering |
  | `SERVICE` | long-running daemon, worker, bot, scheduler |
  | `PIPELINE` | batch data processing (ETL, Airflow DAG, Spark job, ML training) with a clear start and end |
  | `CLI` | command-line entry point with subcommands or arguments |
  | `GUI` | desktop window with widgets |
  | `GAME` | interactive real-time application with a rendering loop |
  | `LIBRARY` | importable package, no end-user interface |

  Examples: `[WEB]`, `[WEB, CLI]`, `[API, SERVICE]`, `[PIPELINE, CLI]`,
  `[GAME, LIBRARY]`.

- **Project Notes**: A short paragraph (≤ 500 chars) capturing details the
  whitelist cannot express — framework choices, special domain
  requirements, anything unique. Examples:

  - `REST API only, no HTML pages — clients are mobile apps`
  - `Discord bot using discord.py, runs as long-lived daemon`
  - `Textual TUI with arrow-key navigation`
  - `Airflow DAG, scheduled daily, reads from S3`

#### 3.3: Merge Background Entries

1. Group Background evidence from all files semantically
2. Generate Description for each group
3. Preserve all original evidence references

#### 3.4: Merge NFR Entries

1. Group NFR evidence from all files by category (Performance, Security, Scalability, etc.)
2. Generate Description for each group
3. Preserve all original evidence references

#### 3.5: Generate feature_spec.md

Create `.rpgkit/data/feature_spec/feature_spec.md`:

```markdown
# Feature Specification

## Meta

- **Repository Name**: {repository_name}
- **Repository Purpose**: {repository_purpose}
- **Project Types**: {comma-separated UPPERCASE tokens, e.g. WEB, CLI}
- **Project Notes**: {short paragraph ≤ 500 chars}
- **Generated At**: {YYYY-MM-DD}
- **Source Documents**: {comma-separated list}

## Background

### BG-001: {Name}
- **Description**: {Description}
- **Evidence**:
  - {ID} | {source} L{line_number}

### BG-002: {Name}
- **Description**: {Description}
- **Evidence**:
  - {ID} | {source} L{line_number}

## NFR

### NFR-001: {Name}
- **Description**: {Description}
- **Evidence**:
  - {ID} | {source} L{line_number}

### NFR-002: {Name}
- **Description**: {Description}
- **Evidence**:
  - {ID} | {source} L{line_number}
```

#### 3.6: Display Progress

```markdown
## ✓ Main File Generation Complete

- **Output File**: .rpgkit/data/feature_spec/feature_spec.md
- **Repository Name**: {name}
- **Background Entries**: <count>
- **NFR Entries**: <count>
```

→ Proceed to **Step 4: Generate Feature Domain Files**

---

### Step 4: Generate Feature Domain Files

Identify feature domains based on FR Evidence and generate detailed feature files.

#### 4.1: Identify Feature Domains

Read original excerpts from all FR evidence and cluster by feature semantics:

**Inference Steps:**

1. **Read original excerpts** - Understand the `> {original excerpt}` content and semantics of each FR
2. **Semantic clustering** - Group functionally related FRs with similar responsibilities into the same feature domain
3. **Name feature domains** - Choose a descriptive name for each cluster

**Clustering Principles:**

- **Functional relevance** - Features within the same domain should be logically closely related
- **Responsibility cohesion** - Features within the same domain should serve the same or similar goals
- **Clear boundaries** - Different domains should have clear responsibility boundaries

**Feature Domain Naming Convention:**

- Use 2-4 English words
- Title Case (capitalize first letter of each word)
- Avoid abbreviations, maintain readability
- Reflect the core responsibility of the domain

#### 4.2: Build Feature Tree Hierarchy

**Build hierarchical structure based on feature semantics from original excerpts:**

1. **Semantic analysis**: Read original excerpts of each FR, understand its functional responsibility
2. **Hierarchy inference**: Determine hierarchy level based on abstraction level and containment relationships
   - High level: General feature descriptions → as parent nodes
   - Low level: Specific feature points → as leaf nodes
3. **Establish relationships**: Group specific features under related abstract features
4. **Reference information**: Chapter structure from Parent Title can serve as reference for hierarchy division, but final decisions are based on feature semantics

#### 4.3: Generate Feature Files

For each domain, create `.rpgkit/data/feature_spec/features/FT-{NNN}.md`:

```markdown
# FT-001: {Domain Name}
- **Description**: {Domain description}

## FT-001-001: {Sub-domain Name}
- **Description**: {Sub-domain description}

### FT-001-001-001: {Feature Name}
- **Description**: {Feature description}
- **Evidence**:
  - {ID} | {source} L{line_number}

### FT-001-001-002: {Feature Name}
- **Description**: {Feature description}
- **Evidence**:
  - {ID} | {source} L{line_number}

## FT-001-002: {Sub-domain Name}
- **Description**: {Sub-domain description}

### FT-001-002-001: {Feature Name}
- **Description**: {Feature description}
- **Evidence**:
  - {ID} | {source} L{line_number}

#### FT-001-002-001-001: {Sub-feature Name}
- **Description**: {Sub-feature description}
- **Evidence**:
  - {ID} | {source} L{line_number}
```

**Hierarchical Structure (determined by heading level):**

| Heading | Level | Description |
| ------- | ----- | ----------- |
| `#` | Domain | FT-001 |
| `##` | Sub-domain | FT-001-001 |
| `###` | Feature | FT-001-001-001 |
| `####` | Sub-feature | FT-001-001-001-001 |
| `#####` | Deeper level | FT-001-001-001-001-001 |
| `######` | Deepest level | FT-001-001-001-001-001-001 |

**Rules:**

- Heading level determines hierarchy depth
- ID follows immediately after heading marker, format is `FT-XXX-XXX-...`
- Number of `-` separated segments in ID = heading level
- Supports up to 6 levels (Markdown heading limit)
- **Hierarchy depth is not fixed** - Determined by actual project needs, not required to use all levels
- **Leaf nodes can be at any level** - For simple projects, leaf nodes may be at level 2 or 3; complex projects may reach level 5 or 6

**Node Types and Evidence:**

| Node Type | Evidence | Description |
| --- | --- | --- |
| **Document nodes** | **Required** | Feature descriptions from documents, regardless of hierarchy level |
| **Organization nodes** | None | Abstract layers created for organizational structure, used to group child nodes |

**Judgment Principles:**

- **Be faithful to documents** - The feature tree should reflect the granularity described in documents
- **Prefer document leaf nodes** - If document descriptions are already precise to specific features (meeting leaf node conditions), adopt them directly
- **Document nodes are leaf nodes** - Feature descriptions from documents serve as leaf nodes (unless the document also describes sub-features)
- **Organization nodes have no Evidence** - Abstract layers inferred for grouping have no direct document source

**Leaf Node Conditions:**

1. **Precise to specific feature** - Describes an independent, identifiable feature point
2. **Not implementation details** - Describes "what" not "how"
3. **Has Evidence support** - Traceable to documents

**Note**: If document-described feature granularity is coarse (not precise to specific features), it still serves as a leaf node at the current stage; subsequent `feature_build` workflow can further refine.

**Evidence Relationships:**

- One node can reference multiple evidence items (same feature described in multiple documents)
- One evidence item corresponds to one node (one feature description corresponds to one feature point)

#### 4.4: Display Progress

```markdown
## ✓ Feature Domain Files Generation Complete

| Domain | File | Feature Count |
|--------|------|---------------|
| FT-001: Core Orchestration | features/FT-001.md | <count> |
| FT-002: User Interface | features/FT-002.md | <count> |
| FT-003: Extensibility | features/FT-003.md | <count> |

**Total Feature Files**: <N>
**Total Feature Count**: <M>
```

→ Proceed to **Step 5: Convert to JSON**

---

### Step 5: Convert to JSON

Convert generated Markdown feature specification files to JSON format.

#### 5.1: Run Conversion Script

Execute the following command:

```bash
rpgkit script feature_spec_to_json.py
```

#### 5.2: Verify Output

Confirm conversion results based on script log output. The script will output logs in a format similar to:

```text
Parsing feature specification from: .rpgkit/data/feature_spec
Include evidence: True

Output written to: .rpgkit/data/feature_spec.json
  - Repository: {name}
  - Background items: <count>
  - NFR items: <count>
  - Top-level features: <count>
  - Total feature nodes: <count>
```

Display results based on log information:

```markdown
## ✓ JSON Conversion Complete

- **Output File**: .rpgkit/data/feature_spec.json
- **Repository**: {from log}
- **Background Entries**: {from log}
- **NFR Entries**: {from log}
- **Top-level Features**: {from log}
- **Total Feature Nodes**: {from log}
```

→ Proceed to **Step 6: Results Report**

---

### Step 6: Results Report

#### 6.1: Summary

```markdown
## Feature Specification Complete

### Output Files

| File | Description |
|------|-------------|
| .rpgkit/data/feature_spec/evidence/*.md | Evidence files |
| .rpgkit/data/feature_spec/feature_spec.md | Main specification file |
| .rpgkit/data/feature_spec/features/FT-*.md | Feature domain files |
| .rpgkit/data/feature_spec.json | JSON format specification file |

### Statistics

| Metric | Count |
|--------|-------|
| Source Documents | <count> |
| Evidence Entries | <count> |
| Background Entries | <count> |
| Feature Domains | <count> |
| Total Features | <count> |
| NFR Entries | <count> |

### Next Steps

To expand and build the feature tree, run:

`/rpgkit.feature_build`
```

---

## Appendix A: Evidence File Format

### Template

```markdown
# Evidence: {document_filename}

## Background / Overview

### [{prefix}-BG-{NNN}] L{start}-L{end}
**Parent Title**: -

> {Original excerpt}

## FR (Functional Requirement)

### [{prefix}-FR-{NNN}] L{start}-L{end}
**Parent Title**: {Section header}

> {Original excerpt}

## NFR (Non-Functional Requirement)

### [{prefix}-NFR-{NNN}] L{start}-L{end}
**Parent Title**: {Section header}

> {Original excerpt}
```

### Example

```markdown
# Evidence: 02_requirements_specification.md

## Background / Overview

### [RS-BG-001] L9-L12
**Parent Title**: -

> This document captures the functional and non-functional requirements for Apache Airflow, a platform to programmatically author, schedule, and monitor workflows.

### [RS-BG-002] L17-L29
**Parent Title**: 2. Target Users and Usage Scenarios

> #### Data Engineer - "Pipeline Builder"
> **Background:** Software engineer specializing in data infrastructure  
> ...

## FR (Functional Requirement)

### [RS-FR-001] L83-L83
**Parent Title**: 3.1 DAG Authoring & Definition

> | FR-DA-001 | Define workflows as Python code using a clear API | Must Have | Core value proposition |

### [RS-FR-002] L84-L84
**Parent Title**: 3.1 DAG Authoring & Definition

> | FR-DA-002 | Express task dependencies explicitly | Must Have | Foundation of scheduling |

### [RS-FR-003] L95-L95
**Parent Title**: 3.2 Task Scheduling

> | FR-TS-001 | Schedule DAG runs based on time intervals (cron-like) | Must Have | Primary scheduling mode |

## NFR (Non-Functional Requirement)

### [RS-NFR-001] L184-L184
**Parent Title**: 4.1 Performance

> | NFR-P-001 | Scheduler latency | < 1 minute from schedule time to task queuing |

### [RS-NFR-002] L194-L194
**Parent Title**: 4.2 Scalability

> | NFR-S-001 | Support horizontal scaling of workers | Distributed execution backends |
```

### Field Descriptions

| Field | Source | Description |
| --- | --- | --- |
| **Parent Title** | Extracted from source | Section header (for tracing source location); `-` for user input mode |

### Line Number Format Specification

All line numbers use unified `L{start}-L{end}` format:

- Single line: `L83-L83`
- Multiple lines: `L21-L24`

---

## Appendix B: Main File Format (feature_spec.md)

### Template

```markdown
# Feature Specification

## Meta

- **Repository Name**: {repository_name}
- **Repository Purpose**: {repository_purpose}
- **Generated At**: {date}
- **Source Documents**: {comma-separated list}

## Background

### BG-{NNN}: {Name}
- **Description**: {Description}
- **Evidence**:
  - {ID} | {source} L{line_number}

## NFR

### NFR-{NNN}: {Name}
- **Description**: {Description}
- **Evidence**:
  - {ID} | {source} L{line_number}
```

### Example

```markdown
# Feature Specification

## Meta

- **Repository Name**: apache-airflow
- **Repository Purpose**: A platform to programmatically author, schedule, and monitor batch-oriented workflows as code with explicit dependencies and reliable execution.
- **Generated At**: 2026-02-05
- **Source Documents**: 01_project_charter.md, 02_requirements_specification.md, 03_domain_analysis.md, 04_system_design_overview.md, 05_interface_data_contract.md, 06_assumptions_constraints_risks.md

## Background

### BG-001: Fragmented Tooling Ecosystem
- **Description**: Organizations currently mix cron jobs, custom scripts, and proprietary scheduling systems without unified workflow definition, execution, and monitoring capabilities.
- **Evidence**:
  - project_charter-BG-001 | 01_project_charter.md L21-L24
  - RS-BG-001 | 02_requirements_specification.md L9-L12

### BG-002: Batch-Oriented Workflow Focus
- **Description**: Apache Airflow focuses on batch-oriented workflow orchestration where work is divided into discrete tasks with clear dependencies.
- **Evidence**:
  - project_charter-BG-002 | 01_project_charter.md L43-L49

## NFR

### NFR-001: Performance
- **Description**: Scheduler latency less than 1 minute from schedule time to task queuing, support 100+ DAGs, UI response under 3 seconds.
- **Evidence**:
  - RS-NFR-001 | 02_requirements_specification.md L184-L184

### NFR-002: Scalability
- **Description**: Support horizontal scaling of workers, distributed execution backends, and multi-scheduler deployment.
- **Evidence**:
  - RS-NFR-002 | 02_requirements_specification.md L194-L194

### NFR-003: Security
- **Description**: Encrypt sensitive data at rest, support authentication/authorization mechanisms, and provide audit logging.
- **Evidence**:
  - RS-NFR-003 | 02_requirements_specification.md L210-L210
```

---

## Appendix C: Feature Domain File Format (features/FT-XXX.md)

### Template

```markdown
# FT-{NNN}: {Domain Name}
- **Description**: {Domain description}

## FT-{NNN}-001: {Sub-domain Name}
- **Description**: {Sub-domain description}

### FT-{NNN}-001-001: {Feature Name}
- **Description**: {Feature description}
- **Evidence**:
  - {ID} | {source} L{line_number}

### FT-{NNN}-001-002: {Feature Name}
- **Description**: {Feature description}
- **Evidence**:
  - {ID} | {source} L{line_number}

## FT-{NNN}-002: {Sub-domain Name}
- **Description**: {Sub-domain description}

### FT-{NNN}-002-001: {Feature Name}
- **Description**: {Feature description}
- **Evidence**:
  - {ID} | {source} L{line_number}
```

### Example

```markdown
# FT-001: Core Orchestration
- **Description**: Core workflow orchestration capabilities including DAG authoring, scheduling, and execution.

## FT-001-001: DAG Authoring and Definition
- **Description**: Enable users to define workflows as Python code with explicit task dependencies.

### FT-001-001-001: Python DAG Definition API
- **Description**: Define workflows using DAG context manager pattern with dag_id, schedule, start_date, default_args, catchup, and tags parameters.
- **Evidence**:
  - project_charter-FR-001 | 01_project_charter.md L116-L118
  - RS-FR-001 | 02_requirements_specification.md L83-L83

### FT-001-001-002: Task Dependency Declaration
- **Description**: Express task dependencies explicitly using >> operator, << operator, chain(), and set_upstream/set_downstream methods.
- **Evidence**:
  - RS-FR-002 | 02_requirements_specification.md L84-L84

### FT-001-001-003: DAG Parameterization
- **Description**: Support parameterized DAGs with Variables, Params, and Jinja templating.
- **Evidence**:
  - RS-FR-003 | 02_requirements_specification.md L85-L85

## FT-001-002: Task Scheduling
- **Description**: Time-based and dependency-based task scheduling capabilities.

### FT-001-002-001: Cron-based Scheduling
- **Description**: Schedule DAG runs based on cron expressions and preset intervals (@daily, @hourly, @weekly, etc.).
- **Evidence**:
  - project_charter-FR-002 | 01_project_charter.md L119-L120
  - RS-FR-004 | 02_requirements_specification.md L95-L95

### FT-001-002-002: Data-aware Scheduling
- **Description**: Trigger DAG runs based on dataset updates and data availability.
- **Evidence**:
  - RS-FR-005 | 02_requirements_specification.md L96-L96

## FT-001-003: Task Execution
- **Description**: Task execution management with multiple backends and reliability features.

### FT-001-003-001: Executor Backends
- **Description**: Support multiple execution backends for different deployment scenarios.

#### FT-001-003-001-001: Local Executor
- **Description**: Execute tasks locally using multiprocessing for development and small deployments.
- **Evidence**:
  - RS-FR-010 | 02_requirements_specification.md L107-L107

#### FT-001-003-001-002: Celery Executor
- **Description**: Distribute task execution across Celery workers for horizontal scaling.
- **Evidence**:
  - SDO-FR-005 | 04_system_design_overview.md L89-L89

#### FT-001-003-001-003: Kubernetes Executor
- **Description**: Execute each task in a separate Kubernetes pod for isolation and scalability.
- **Evidence**:
  - SDO-FR-006 | 04_system_design_overview.md L91-L91

### FT-001-003-002: Retry and Failure Handling
- **Description**: Automatic task retry with configurable attempts, delays, and exponential backoff.
- **Evidence**:
  - RS-FR-011 | 02_requirements_specification.md L108-L108
```

---

## Evidence ID Format Description

### ID Structure

```text
{file_prefix}-{category}-{sequence}
     │           │         │
     │           │         └── 001, 002, 003... (numbered independently per category)
     │           └────── BG / FR / NFR
     └──────────── File name abbreviation or full name
```

### Category Meanings

| Category | Full Name | Description |
| --- | --- | --- |
| BG | Background | Background information, context, goals, terminology definitions |
| FR | Functional Requirement | Functional requirements, features, behaviors, operations |
| NFR | Non-Functional Requirement | Performance, security, scalability, assumptions, constraints, technical risks |

### Examples

| ID | Interpretation |
| --- | --- |
| `RS-FR-001` | requirements_specification - FR - 1st entry |
| `RS-NFR-003` | requirements_specification - NFR - 3rd entry |
| `project_charter-BG-001` | project_charter - Background - 1st entry |
| `IDC-FR-004` | interface_data_contract - FR - 4th entry |
| `user_input-FR-001` | user_input - FR - 1st entry |

---

## Quality Standards

### Evidence Quality

- [ ] Original text preserved verbatim
- [ ] Line numbers accurately traceable
- [ ] Each evidence entry has unique ID (format: `{prefix}-{BG|FR|NFR}-{sequence}`)
- [ ] Correctly categorized (Background / FR / NFR)
- [ ] Summary is feature description/summary (extracted from source or generated based on content)
- [ ] Parent Title is section header from source (`-` for user input)

### Feature Tree Quality

- [ ] Hierarchical structure is logically consistent
- [ ] Numbering follows decimal outline format
- [ ] ID matches numbering structure
- [ ] Document nodes (feature descriptions from documents) have Evidence
- [ ] Organization nodes (abstract layers for grouping) have no Evidence
- [ ] Each document node has clear feature boundaries (describable in one sentence)
- [ ] Node descriptions say "what" not "how"
- [ ] No orphaned or duplicate features

### File Quality

- [ ] All files saved to correct locations
- [ ] File links in feature_spec.md are valid
- [ ] Markdown formatting is correct
- [ ] No broken references between files

```text
