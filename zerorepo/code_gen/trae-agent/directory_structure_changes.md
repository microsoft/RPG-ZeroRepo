# Directory Structure Changes Summary

This document summarizes the changes made to reorganize the directory structure in `main.py` to better separate the workspace from the trae-agent source code.

## Previous Structure

Previously, when `working_dir` was set to `"./traeagent_workspace"`, the structure was:
```
. (parent directory)
├── traeagent_workspace/     (working_dir - user's project files)
├── results/                 (results_dir - task results)
└── trae-agent/             (trae_agent_dir - trae-agent artifacts)
```

## New Structure

Now, when `working_dir` is set to `"./traeagent_workspace"`, the structure is:
```
./traeagent_workspace/       (working_dir - user's project files)
├── results/                 (results_dir - task results)
└── .trae-agent-artifacts/   (trae_agent_dir - trae-agent build artifacts)
```

## Key Changes

1. **Directory Organization**:
   - `results_dir` is now `working_dir / "results"` instead of `working_dir.parent / "results"`
   - `trae_agent_dir` is now `working_dir / ".trae-agent-artifacts"` instead of `working_dir.parent / "trae-agent"`
   - All workspace-related files are now contained within the working directory

2. **Docker Container Structure**:
   - `/trae-workspace`: Contains only the user's project files (mounted from `working_dir`)
   - `/trae-src`: Contains the trae-agent source code (mounted read-only from the repository)
   - `/opt/trae-agent`: Where trae-agent is installed in the container (not in workspace)
   - `/tmp/trae-agent-build`: Temporary directory used during build process

3. **Artifact Storage**:
   - Build artifacts (tar files) are stored in `.trae-agent-artifacts/` within the working directory
   - Config file is copied to `.trae-agent-artifacts/` and then transferred to the container
   - Keeps the user's workspace root clean

4. **Container Setup**:
   - Trae-agent is built in `/tmp/trae-agent-build` and installed to `/opt/trae-agent`
   - This prevents trae-agent files from cluttering the workspace
   - The workspace at `/trae-workspace` contains only the user's project files

## Benefits

1. **Cleaner Separation**: The workspace contains only user project files, not trae-agent internals
2. **Self-Contained**: All workspace-related files (including results and artifacts) are within the working directory
3. **No Parent Directory Access**: Eliminates the need to access parent directories, making the structure more predictable
4. **Hidden Artifacts**: Using `.trae-agent-artifacts` keeps build artifacts hidden from normal directory listings

## Usage

No changes are required to the API. Users can continue to specify `working_dir` as before:

```python
runner = ContinuousTraeAgentRunner(
    working_dir="./my_workspace",  # Everything will be contained within this directory
    # ... other parameters
)
```