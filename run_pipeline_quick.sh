#!/usr/bin/env bash
# Quick/dev variant of run_pipeline.sh: small subsets, fewer MF-pretraining
# iterations/folds/dims, and fewer PPO epochs/cell-lines.
#
# All quick-pipeline artifacts (CV folds, MF weights, results, checkpoints) are
# written under a separate data/GDSC_ALL_QUICK tree, so this never touches the
# full data/GDSC_ALL CV folds, models, or results from run_pipeline.sh.
#
# NOT meant for reported results — only for fast pipeline sanity checks / iteration.
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python"
fi

SRC_DATA_DIR="data/GDSC_ALL"
DATA_DIR="data/GDSC_ALL_QUICK"
# NOTE: keep F >= deep_out_size+4 (now 32+4=36) — smaller values trigger a shape
# mismatch in the critic's ConvValueNet pooling (see configG_FULL_quick.yaml).
# This quick run mirrors the paper-faithful PPO flags of run_pipeline.sh
# (K=8, gamma=0.95, c2=0.001, 2 cross layers, deep 128->64->32) so it can act as a
# smoke-test for deep_out_size=32 + the Pearson-kernel features before the full run.
NUM_PROCESSES=32    # GPU run: larger batches are ~free, so use them to cut batches/epoch
F=100               # projection dimension (matches full pipeline)
ALGO=ppo
ANALYSIS=FULL
NFOLDS=2            # CV folds (full pipeline: 5)
ITERS=1000          # MF pretraining iterations per fold (full pipeline: 20000) — CPU-only, unaffected by GPU
EPOCHS=30           # PPO training epochs (full pipeline: 1400)
PROP=0.5            # fraction of training cell-lines used per fold (full pipeline: 1.0)

# Optional: set GPU device index (used only if CUDA is available)
CUDA_ID="${CUDA_ID:-0}"

# ── GPU detection ────────────────────────────────────────────────────────────
GPU_STATUS=$("$PYTHON" - <<'EOF'
import torch
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"CUDA available — {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda}, {n} device(s))")
else:
    print("No GPU detected — running on CPU")
EOF
)

LOG_FILE="pipeline_quick_$(date +%Y%m%d_%H%M%S).log"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_FILE"; }
elapsed() { echo $(( $(date +%s) - $1 )); }

log "======================================================"
log "PPORank QUICK pipeline — $(date)"
log "Python  : $PYTHON"
log "GPU     : $GPU_STATUS"
log "Data    : $DATA_DIR"
log "Folds   : $NFOLDS | f=$F | MF iters=$ITERS | epochs=$EPOCHS | prop=$PROP | num_processes=$NUM_PROCESSES"
log "CUDA ID : $CUDA_ID"
log "Log     : $LOG_FILE"
log "======================================================"

TOTAL_START=$(date +%s)

# ── Step 1: Preprocess raw GDSC files → data/GDSC_ALL/GDSC_GEX.npz ──────────
# (load_dataset.py always writes to data/GDSC_ALL — shared with the full pipeline)
log ""
log ">>> STEP 1: Preprocessing raw GDSC files"
STEP_START=$(date +%s)

if [ -f "$SRC_DATA_DIR/GDSC_GEX.npz" ]; then
    log "    GDSC_GEX.npz already exists — skipping Step 1"
else
    "$PYTHON" preprocess/load_dataset.py preprocess/load_GDSC.txt 2>&1 | tee -a "$LOG_FILE"
    log "    Step 1 done in $(elapsed $STEP_START)s"
fi

# ── Step 1b: Link the shared raw inputs into the quick data dir ─────────────
log ""
log ">>> STEP 1b: Linking raw inputs into $DATA_DIR"
mkdir -p "$DATA_DIR"
for fname in GDSC_GEX.npz gdsc_697_genes.csv; do
    if [ ! -e "$DATA_DIR/$fname" ]; then
        ln "$SRC_DATA_DIR/$fname" "$DATA_DIR/$fname"
        log "    linked $fname"
    fi
done

# ── Step 2: small CV split + kernel features + MF layer pretraining ─────────
log ""
log ">>> STEP 2: CV split + kernel features + MF layer pretraining (quick)"
STEP_START=$(date +%s)

FOLD0_CHECK="$DATA_DIR/CV/FULL/Fold0/Xtrain_rawDf.csv"
if [ -f "$FOLD0_CHECK" ]; then
    log "    CV folds already exist — checking MF pretraining..."
    FOLD0_MF="$DATA_DIR/CV/FULL/Fold0/${F}Dim/WPmatrix.csv"
    if [ -f "$FOLD0_MF" ]; then
        log "    MF pretrained weights already exist — skipping Step 2"
    else
        log "    Folds exist but MF weights missing — re-running --decompose only"
        "$PYTHON" prepare.py --decompose --data_dir "$DATA_DIR" --config configs/configG_FULL_quick.yaml \
            --nfolds "$NFOLDS" --f "$F" --iters "$ITERS" 2>&1 | tee -a "$LOG_FILE"
        log "    Step 2 (decompose only) done in $(elapsed $STEP_START)s"
    fi
else
    "$PYTHON" prepare.py --decompose --data_dir "$DATA_DIR" --config configs/configG_FULL_quick.yaml \
        --nfolds "$NFOLDS" --f "$F" --iters "$ITERS" 2>&1 | tee -a "$LOG_FILE"
    log "    Step 2 done in $(elapsed $STEP_START)s"
fi

# ── Step 3: Train PPORank for each fold (quick) ──────────────────────────────
log ""
log ">>> STEP 3: Training PPORank ($NFOLDS folds, $EPOCHS epochs each)"

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

    # Same paper-faithful PPO flags as run_pipeline.sh (§3.1.3), so this is a true
    # smoke-test of that config. num_processes here is the quick value (32), not
    # the paper's 16 — fine for a sanity check, not for reported numbers.
    "$PYTHON" main.py \
        --num_processes "$NUM_PROCESSES" \
        --Data "$DATA_DIR" \
        --analysis "$ANALYSIS" \
        --algo "$ALGO" \
        --f "$F" \
        --epochs "$EPOCHS" \
        --prop "$PROP" \
        --normalize_y \
        --fold "$FOLD" \
        --cuda_id "$CUDA_ID" \
        --ppo_epoch 8 \
        --gamma 0.95 \
        --entropy_coef 0.001 \
        --nlayers_cross 2 \
        --deep_hidden_sizes 128 64 \
        --deep_out_size 32 \
        2>&1 | tee -a "$LOG_FILE"

    log "    $FOLD done in $(elapsed $FOLD_START)s"
done

# ── Step 4: Aggregate results ────────────────────────────────────────────────
log ""
log ">>> STEP 4: Aggregating results"
STEP_START=$(date +%s)

"$PYTHON" results.py --config ./configs/configG_FULL_quick.yaml 2>&1 | tee results_ppo_quick.txt
log "    Results written to results_ppo_quick.txt  [$(elapsed $STEP_START)s]"

# ── Done ─────────────────────────────────────────────────────────────────────
log ""
log "======================================================"
log "Quick pipeline complete — total elapsed: $(elapsed $TOTAL_START)s"
log "Results : results_ppo_quick.txt"
log "Logs    : $LOG_FILE"
log "======================================================"
