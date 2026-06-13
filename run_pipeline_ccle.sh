#!/usr/bin/env bash
# CCLE pipeline: prepare → train (5-fold CV) → baselines → aggregate
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"

DATA_DIR="data/CCLE"
NUM_PROCESSES=8                # CCLE: paper uses 8 parallel actors
F=100
ALGO=ppo
ANALYSIS=FULL
NFOLDS=5
SOURCE=CCLE                    # needed for Data_All=False file names

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CUDA_ID="${CUDA_ID:-0}"
GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.47}"
MEM_FLAG=""
[ -n "$GPU_MEM_FRACTION" ] && MEM_FLAG="--gpu_mem_fraction $GPU_MEM_FRACTION"

LOG_FILE="pipeline_ccle_$(date +%Y%m%d_%H%M%S).log"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_FILE"; }
elapsed() { echo $(( $(date +%s) - $1 )); }

log "======================================================"
log "CCLE PPORank pipeline — $(date)"
log "Python  : $PYTHON"
log "Data    : $DATA_DIR"
log "Folds   : $NFOLDS"
log "CUDA ID : $CUDA_ID"
log "Log     : $LOG_FILE"
log "======================================================"

TOTAL_START=$(date +%s)

# ── Step 1: CV split + kernel CSVs + MF pretraining ────────────────────
log ""
log ">>> STEP 1: CV split + MF pretraining"
STEP_START=$(date +%s)

FOLD0_CHECK="$DATA_DIR/CV/FULL/Fold0/Xtrain_rawDf.csv"
if [ -f "$FOLD0_CHECK" ]; then
    log "    CV folds already exist — checking MF pretraining..."
    FOLD0_MF="$DATA_DIR/CV/FULL/Fold0/${F}Dim/WPmatrix.csv"
    if [ -f "$FOLD0_MF" ]; then
        log "    MF weights already exist — skipping Step 1"
    else
        log "    Folds exist but MF missing — running --decompose only"
        # CCLE has only 19 drugs / ~390 cells, so the MF (CaDRReS) gradient step
        # ~1/n_K is much larger than on GDSC; lr=0.01 diverges to NaN (degenerate
        # WP, loss prints 0.000000, PPO then gets NaN logits). lr=0.001 stabilises it.
        "$PYTHON" prepare.py --data_dir "$DATA_DIR" --Source "$SOURCE" \
            --decompose --lr 0.001 --config configs/configC_FULL_compare.yaml
        log "    Step 1 (decompose only) done in $(elapsed $STEP_START)s"
    fi
else
    # lr=0.001 (vs default 0.01): CCLE's small problem (19 drugs) makes the MF
    # gradient step too large at 0.01 and it diverges to NaN — see note above.
    "$PYTHON" prepare.py --data_dir "$DATA_DIR" --Source "$SOURCE" \
        --decompose --lr 0.001 --config configs/configC_FULL_compare.yaml
    log "    Step 1 done in $(elapsed $STEP_START)s"
fi

# ── Step 2: Train PPORank for each fold ────────────────────────────────
log ""
log ">>> STEP 2: Training PPORank ($NFOLDS folds)"

for FOLD_IDX in $(seq 0 $(( NFOLDS - 1 ))); do
    FOLD="Fold${FOLD_IDX}"
    RESULT_CHECK="results/$DATA_DIR/FULL/${F}Dim/ppo/ppo_${FOLD_IDX}.npz"
    log ""
    log "  --- $FOLD ---"
    FOLD_START=$(date +%s)

    if [ -f "$RESULT_CHECK" ]; then
        log "    Result already exists ($RESULT_CHECK) — skipping"
        continue
    fi

    "$PYTHON" main.py \
        --num_processes "$NUM_PROCESSES" \
        --Data "$DATA_DIR" \
        --analysis "$ANALYSIS" \
        --algo "$ALGO" \
        --f "$F" \
        --normalize_y \
        --fold "$FOLD" \
        --cuda_id "$CUDA_ID" \
        --ppo_epoch 8 \
        --gamma 0.95 \
        --entropy_coef 0.001 \
        --nlayers_cross 2 \
        --deep_hidden_sizes 128 64 \
        --deep_out_size 32 \
        $MEM_FLAG \
        2>&1 | tee -a "$LOG_FILE"

    log "    $FOLD done in $(elapsed $FOLD_START)s"
done

# ── Step 3: Baselines (EN, KRR) ────────────────────────────────────────
log ""
log ">>> STEP 3: Baselines (EN, KRR)"
STEP_START=$(date +%s)
"$PYTHON" baselines.py --config configs/configC_FULL_compare.yaml \
    --methods EN KRR --overwrite 2>&1 | tee -a "$LOG_FILE"
log "    Baselines done in $(elapsed $STEP_START)s"

# ── Step 4: KRL (CPU, can run alongside anything) ──────────────────────
log ""
log ">>> STEP 4: KRL baseline"
STEP_START=$(date +%s)
"$PYTHON" baselines_krl.py --config configs/configC_FULL_compare.yaml \
    --overwrite 2>&1 | tee -a "$LOG_FILE"
log "    KRL done in $(elapsed $STEP_START)s"

# ── Step 5: Aggregate results ──────────────────────────────────────────
log ""
log ">>> STEP 5: Aggregating results"
STEP_START=$(date +%s)
"$PYTHON" results.py --config configs/configC_FULL_compare.yaml \
    2>&1 | tee "$DATA_DIR/../results_ccle.txt"
log "    Results written to results_ccle.txt  [$(elapsed $STEP_START)s]"

# ── Done ───────────────────────────────────────────────────────────────
log ""
log "======================================================"
log "Pipeline complete — total elapsed: $(elapsed $TOTAL_START)s"
log "Results : results_ccle.txt"
log "Logs    : $LOG_FILE"
log "======================================================"