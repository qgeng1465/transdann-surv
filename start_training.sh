#!/usr/bin/env bash
# =============================================================================
# start_training.sh — Launch the full DANN training pipeline in background
# =============================================================================
# Run this, then come back in a few hours to check results.
#
# Usage:
#   bash start_training.sh
#   tail -f results/logs/training.log
#
# After training completes:
#   python3 scripts/03_visualize_lihc.py   # generate visualizations
#   ls -la results/figures/lihc/           # see all figures
# =============================================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Make the repo root importable (e.g., scripts.paths, scripts.transdann_utils).
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

echo "======================================================================"
echo "  TransDANN LIHC Training Pipeline"
echo "  Started: $(date)"
echo "  GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'No GPU')"
echo "  PID: $$"
echo "======================================================================"
echo ""
echo "  Training 3 experiments (A: TCGA vs SEER, B: TCGA vs External, C: All)"
echo "  Each experiment: DANN (with GRL) + Baseline (no GRL)"
echo "  200 epochs, DeepHit survival head"
echo ""
echo "  To check progress: tail -f results/logs/training.log"
echo "  To check GPU: watch -n 5 nvidia-smi"
echo "======================================================================"

# Run training — this is the long part (3-6 hours)
mkdir -p results/logs
python3 scripts/02_train_dann_lihc.py \
    --experiment ALL \
    --epochs 200 \
    --surv_type deephit \
    --d_model 128 \
    --n_layers 4 \
    --dropout 0.15 \
    --lr 5e-4 \
    --domain_weight 0.3 \
    2>&1 | tee results/logs/training.log

EXIT_CODE=$?
echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  TRAINING COMPLETE! ($(date))"
else
    echo "  TRAINING FINISHED WITH ERRORS (exit code $EXIT_CODE) at $(date)"
fi
echo "======================================================================"
echo ""
echo "  Generating visualizations..."
python3 scripts/03_visualize_lihc.py 2>&1 | tee -a results/logs/training.log

echo ""
echo "======================================================================"
echo "  ALL DONE! ($(date))"
echo "======================================================================"
echo "  Results:"
echo "    - Training logs:   results/logs/training.log"
echo "    - Model results:   results/lihc_experiments/"
echo "    - Figures:         results/figures/lihc/"
echo "    - Summary CSV:     results/figures/lihc/results_summary.csv"
echo "======================================================================"
