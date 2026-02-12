#!/bin/bash
set -euo pipefail

# ============================================================================
# ZeroRepo Main Pipeline Runner (fixed-args version)
# Edit the variables below to match your setup.
# ============================================================================

# ---------- User-editable settings ----------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GENERATING_PROJECT="../eat_snake"

# Azure AD authentication (for Azure OpenAI token-based auth)
export AZURE_TENANT_ID="${AZURE_TENANT_ID:-72f988bf-86f1-41af-91ab-2d7cd011db47}"
export AZURE_TOKEN_SCOPE="${AZURE_TOKEN_SCOPE:-api://feb7b661-cac7-44a8-8dc1-163b63c23df2/.default}"

CONFIG="${PROJECT_ROOT}/configs/zerorepo_config.yaml"

CHECKPOINT_DIR="${GENERATING_PROJECT}/checkpoints"
REPO_DIR="${GENERATING_PROJECT}/workspace"


PHASE="all"          # all | design | implementation
LOG_LEVEL="INFO"     # DEBUG | INFO | WARNING | ERROR
LOG_FILE="${CHECKPOINT_DIR}/zerorepo.log"

LLM_CONFIG=""        # e.g., "${PROJECT_ROOT}/configs/llm.yaml" (leave empty to disable)
FORCE_REBUILD=false  # true | false
RESUME=true         # true | false
DRY_RUN=false        # true | false

# ---------- Build command ----------
CMD=(
  python "${PROJECT_ROOT}/main.py"
  --config "${CONFIG}"
  --checkpoint "${CHECKPOINT_DIR}"
  --repo "${REPO_DIR}"
  --phase "${PHASE}"
  --log-level "${LOG_LEVEL}"
  --log-file "${LOG_FILE}"
)

if [[ -n "${LLM_CONFIG}" ]]; then
  CMD+=(--llm-config "${LLM_CONFIG}")
fi
if [[ "${FORCE_REBUILD}" == "true" ]]; then
  CMD+=(--force-rebuild)
fi
if [[ "${RESUME}" == "true" ]]; then
  CMD+=(--resume)
fi
if [[ "${DRY_RUN}" == "true" ]]; then
  CMD+=(--dry-run)
fi

# ---------- Run ----------
echo "=========================================="
echo " ZeroRepo Pipeline"
echo "=========================================="
echo " Config:     ${CONFIG}"
echo " Checkpoint: ${CHECKPOINT_DIR}"
echo " Repo:       ${REPO_DIR}"
echo " Phase:      ${PHASE}"
echo " Log Level:  ${LOG_LEVEL}"
echo " Log File:   ${LOG_FILE}"
echo " LLM Config: ${LLM_CONFIG:-<none>}"
echo " Rebuild:    ${FORCE_REBUILD}"
echo " Resume:     ${RESUME}"
echo " Dry Run:    ${DRY_RUN}"
echo "=========================================="

mkdir -p "${CHECKPOINT_DIR}"
exec "${CMD[@]}"