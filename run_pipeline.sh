#!/usr/bin/env bash
# Full PPORank pipeline: preprocess → prepare → train (5-fold CV) → evaluate
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python"
fi

DATA_DIR="data/GDSC_ALL"
NUM_PROCESSES=16
F=100               # projection dimension
ALGO=ppo
ANALYSIS=FULL
NFOLDS=5

# Optional: set GPU device index (used only if CUDA is available)
CUDA_ID="${CUDA_ID:-0}"

LOG_FILE="pipeline_$(date +%Y%m%d_%H%M%S).log"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_FILE"; }
elapsed() { echo $(( $(date +%s) - $1 )); }

log "======================================================"
log "PPORank pipeline — $(date)"
log "Python  : $PYTHON"
log "Data    : $DATA_DIR"
log "Folds   : $NFOLDS"
log "CUDA ID : $CUDA_ID"
log "Log     : $LOG_FILE"
log "======================================================"

TOTAL_START=$(date +%s)

# ── Step 1: Preprocess raw GDSC files → data/GDSC_ALL/GDSC_GEX.npz ──────────
log ""
log ">>> STEP 1: Preprocessing raw GDSC files"
STEP_START=$(date +%s)

if [ -f "$DATA_DIR/GDSC_GEX.npz" ]; then
    log "    GDSC_GEX.npz already exists — skipping Step 1"
else
    $PYTHON preprocess/load_dataset.py preprocess/load_GDSC.txt 2>&1 | tee -a "$LOG_FILE"
    log "    Step 1 done in $(elapsed $STEP_START)s"
fi

# ── Step 2: 5-fold CV split + Pearson kernel features + MF pretraining ───────
log ""
log ">>> STEP 2: CV split + kernel features + MF layer pretraining"
STEP_START=$(date +%s)

FOLD0_CHECK="$DATA_DIR/CV/FULL/Fold0/Xtrain_rawDf.csv"
if [ -f "$FOLD0_CHECK" ]; then
    log "    CV folds already exist — checking MF pretraining..."
    FOLD0_MF="$DATA_DIR/CV/FULL/Fold0/${F}Dim/WPmatrix.csv"
    if [ -f "$FOLD0_MF" ]; then
        log "    MF pretrained weights already exist — skipping Step 2"
    else
        log "    Folds exist but MF weights missing — re-running --decompose only"
        $PYTHON prepare.py --decompose 2>&1 | tee -a "$LOG_FILE"
        log "    Step 2 (decompose only) done in $(elapsed $STEP_START)s"
    fi
else
    $PYTHON prepare.py --decompose 2>&1 | tee -a "$LOG_FILE"
    log "    Step 2 done in $(elapsed $STEP_START)s"
fi

# ── Step 3: Train PPORank for each fold ──────────────────────────────────────
log ""
log ">>> STEP 3: Training PPORank ($NFOLDS folds)"

for FOLD_IDX in $(seq 0 $(( NFOLDS - 1 ))); do
    FOLD="Fold${FOLD_IDX}"
    RESULT_CHECK="results/$DATA_DIR/FULL/${F}Dim/ppo/ppo_${FOLD_IDX}.npz"

    log ""
    log "  --- $FOLD ---"
    FOLD_START=$(date +%s)

    if [ -f "$RESULT_CHECK" ]; then
        log "    Result already exists ($RESULT_CHECK) — skipping $FOLD"
        continue
    fi

    $PYTHON main.py \
        --num_processes "$NUM_PROCESSES" \
        --Data "$DATA_DIR" \
        --analysis "$ANALYSIS" \
        --algo "$ALGO" \
        --f "$F" \
        --normalize_y \
        --fold "$FOLD" \
        --cuda_id "$CUDA_ID" \
        2>&1 | tee -a "$LOG_FILE"

    log "    $FOLD done in $(elapsed $FOLD_START)s"
done

# ── Step 4: Aggregate results ────────────────────────────────────────────────
log ""
log ">>> STEP 4: Aggregating results"
STEP_START=$(date +%s)

$PYTHON results.py --config ./configs/configG_FULL.yaml 2>&1 | tee results_ppo.txt
log "    Results written to results_ppo.txt  [$(elapsed $STEP_START)s]"

# ── Done ─────────────────────────────────────────────────────────────────────
log ""
log "======================================================"
log "Pipeline complete — total elapsed: $(elapsed $TOTAL_START)s"
log "Results : results_ppo.txt"
log "Logs    : $LOG_FILE"
log "======================================================"
