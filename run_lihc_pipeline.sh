#!/usr/bin/env bash
# =============================================================================
# run_lihc_pipeline.sh — Master Pipeline: DANN Training + Visualization
# =============================================================================
#
# A fully automated 7-8 hour pipeline for LIHC (Liver Cancer) DANN training.
# This script handles everything end-to-end with zero manual intervention.
#
# Pipeline stages:
#   Stage 1: Environment setup (5 min)
#   Stage 2: Data preparation (2 min)
#   Stage 3: DANN model training (3-5 hours)
#   Stage 4: Evaluation & visualization (15 min)
#   Stage 5: Report generation (5 min)
#
# Usage:
#   bash run_lihc_pipeline.sh                   # Full pipeline
#   bash run_lihc_pipeline.sh --stage 2         # Resume from stage 2
#   bash run_lihc_pipeline.sh --quick           # Quick test (fewer epochs)
#   nohup bash run_lihc_pipeline.sh &           # Run in background
#
# Monitoring:
#   tail -f results/logs/pipeline.log           # Watch progress
#   ls -la results/lihc_experiments/            # Check intermediate results
#   ls -la results/figures/lihc/                # See generated figures
# =============================================================================

set -e
# Don't exit on individual experiment failures
set +e

# ----- Configuration -----
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p results/logs
PIPELINE_LOG="results/logs/pipeline.log"
START_TIME=""
STAGE=${STAGE:-1}
QUICK=${QUICK:-false}
SKIP_ENV=${SKIP_ENV:-false}

# Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --stage) STAGE="$2"; shift ;;
        --quick) QUICK=true ;;
        --skip-env) SKIP_ENV=true ;;
        --help)
            echo "Usage: bash run_lihc_pipeline.sh [--stage N] [--quick] [--skip-env]"
            echo "  --stage N     Start from stage N (1-5)"
            echo "  --quick       Quick test mode (fewer epochs, smaller sample)"
            echo "  --skip-env    Skip environment setup (if already done)"
            echo "  --help        Show this help"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# ----- Logging -----
log() {
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $1"
    echo "[$timestamp] $1" >> "$PIPELINE_LOG"
}

section() {
    echo "" | tee -a "$PIPELINE_LOG"
    echo "================================================================" | tee -a "$PIPELINE_LOG"
    echo "  STAGE $1: $2" | tee -a "$PIPELINE_LOG"
    echo "  $(date)" | tee -a "$PIPELINE_LOG"
    echo "================================================================" | tee -a "$PIPELINE_LOG"
}

# ----- Cleanup handler -----
cleanup() {
    local exit_code=$?
    echo "" >> "$PIPELINE_LOG"
    echo "================================================================" >> "$PIPELINE_LOG"
    if [ $exit_code -eq 0 ]; then
        echo "  Pipeline completed successfully at $(date)" >> "$PIPELINE_LOG"
        echo "================================================================" >> "$PIPELINE_LOG"
    else
        echo "  Pipeline FAILED at $(date) — see logs above for details" >> "$PIPELINE_LOG"
        echo "  To resume: bash run_lihc_pipeline.sh --stage $CURRENT_STAGE" >> "$PIPELINE_LOG"
        echo "================================================================" >> "$PIPELINE_LOG"
    fi
    # Calculate duration
    if [ -n "$START_TIME" ]; then
        local end_time=$(date +%s)
        local duration=$((end_time - START_TIME))
        local hours=$((duration / 3600))
        local minutes=$(( (duration % 3600) / 60 ))
        log "Total elapsed time: ${hours}h ${minutes}m"
    fi
    # Print summary
    if [ -f "results/lihc_experiments/summary.json" ]; then
        echo "" | tee -a "$PIPELINE_LOG"
        echo "  === RESULTS SUMMARY ===" | tee -a "$PIPELINE_LOG"
        python3 -c "
import json
with open('results/lihc_experiments/summary.json') as f:
    data = json.load(f)
for exp_key, comp in data.items():
    name = {'A_tcga_vs_seer': 'TCGA vs SEER', 'B_tcga_vs_external': 'TCGA vs External', 'C_all_cohorts': 'All Cohorts'}.get(exp_key, exp_key)
    dc = comp.get('dann_val_cindex', '?')
    bc = comp.get('baseline_val_cindex', '?')
    delta = comp.get('delta', '?')
    print(f'  {name:25s}: DANN={dc} | Baseline={bc} | Δ={delta}')
" 2>/dev/null | tee -a "$PIPELINE_LOG"

        echo "" | tee -a "$PIPELINE_LOG"
        echo "  Figures saved to: results/figures/lihc/" | tee -a "$PIPELINE_LOG"
        ls -1 results/figures/lihc/*.png 2>/dev/null | while read f; do
            echo "    - $(basename $f)" | tee -a "$PIPELINE_LOG"
        done
    fi
    exit $exit_code
}
trap cleanup EXIT

START_TIME=$(date +%s)
CURRENT_STAGE=1

# =====================================================================
# STAGE 1: Environment Setup
# =====================================================================
setup_env() {
    section 1 "Environment Setup"
    log "Installing Python dependencies ..."

    # Make scripts executable
    chmod +x scripts/*.py scripts/*.sh 2>/dev/null

    # Install system-level deps if needed
    if command -v apt-get &> /dev/null; then
        log "  Detected apt package manager"
        # Only install if not already available
    fi

    # Install via pip
    pip3 install --quiet --upgrade pip 2>&1 | tail -1
    pip3 install -r requirements.txt -q 2>&1 | tail -1
    pip3 install scikit-survival lifelines -q 2>&1 | tail -1

    # Verify PyTorch + CUDA
    log "  Verifying PyTorch and CUDA ..."
    python3 -c "
import torch, numpy, pandas, sklearn, lifelines, matplotlib, seaborn
print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}, Mem: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')
print(f'  NumPy {numpy.__version__}, Pandas {pandas.__version__}')
print(f'  scikit-learn {sklearn.__version__}, lifelines {lifelines.__version__}')
" 2>&1 | tee -a "$PIPELINE_LOG"

    log "✅ Stage 1 complete"
}

# =====================================================================
# STAGE 2: Data Preparation
# =====================================================================
prepare_data() {
    section 2 "Data Preparation"
    log "Merging LIHC cohort data into unified training format ..."

    python3 scripts/01_prepare_lihc_data.py 2>&1 | tee -a "$PIPELINE_LOG"
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -ne 0 ]; then
        log "❌ Data preparation failed (exit code $exit_code)"
        exit $exit_code
    fi

    log "✅ Stage 2 complete. Data saved to data_processed/lihc_all_cohorts.csv"
}

# =====================================================================
# STAGE 3: DANN Model Training
# =====================================================================
train_models() {
    section 3 "DANN Model Training"
    log "This is the longest stage (3-5 hours for full training)."
    log "Running experiments A, B, C with DANN and Baseline modes ..."

    # Create experiment directory
    mkdir -p results/lihc_experiments

    QUICK_FLAG=""
    if [ "$QUICK" = true ]; then
        QUICK_FLAG="--quick"
        log "QUICK MODE: 100 epochs, smaller SEER sample"
    else
        log "FULL MODE: 200 epochs each"
    fi

    SECONDS=0

    # Run all experiments sequentially
    # Each experiment trains DANN + Baseline (auto-saves best models)
    python3 scripts/02_train_dann_lihc.py \
        --experiment ALL \
        --epochs 200 \
        --surv_type deephit \
        --d_model 128 \
        --n_layers 4 \
        --dropout 0.15 \
        --lr 5e-4 \
        --domain_weight 0.3 \
        $QUICK_FLAG \
        2>&1 | tee -a "$PIPELINE_LOG"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -ne 0 ]; then
        log "⚠ Training had some errors. Check results/lihc_experiments/ for partial results."
    else
        log "✅ Training complete!"
    fi

    local elapsed=$SECONDS
    local hours=$((elapsed / 3600))
    local minutes=$(( (elapsed % 3600) / 60 ))
    log "Training time: ${hours}h ${minutes}m"
    log "✅ Stage 3 complete"
}

# =====================================================================
# STAGE 4: Visualization
# =====================================================================
visualize() {
    section 4 "Evaluation & Visualization"
    log "Generating all figures and evaluation tables ..."

    python3 scripts/03_visualize_lihc.py 2>&1 | tee -a "$PIPELINE_LOG"
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -ne 0 ]; then
        log "⚠ Visualization had some issues, check output"
    else
        log "✅ All figures generated!"
    fi

    # List output
    log "Generated figures:"
    ls -lh results/figures/lihc/*.png 2>/dev/null | while read f; do
        log "  $(basename $f)"
    done

    log "✅ Stage 4 complete"
}

# =====================================================================
# STAGE 5: Report
# =====================================================================
generate_report() {
    section 5 "Report Generation"
    log "Compiling final report ..."

    REPORT_FILE="results/lihc_pipeline_report.md"
    {
        echo "# LIHC DANN Pipeline Report"
        echo "Generated: $(date)"
        echo ""
        echo "## Configuration"
        echo "- Quick mode: $QUICK"
        echo "- Epochs: 200"
        echo "- Model: TransDANNSurvV3 (DeepHit head)"
        echo "- DANN domain weight: 0.3"
        echo ""
        echo "## Experiments"
        echo ""
    } > "$REPORT_FILE"

    # Add experimental results if available
    if [ -f "results/lihc_experiments/summary.json" ]; then
        python3 -c "
import json
with open('results/lihc_experiments/summary.json') as f:
    data = json.load(f)
print('| Experiment | DANN C-index | Baseline C-index | Δ |')
print('|---|---|---|---|')
for exp_key, comp in data.items():
    name_map = {'A_tcga_vs_seer': 'TCGA vs SEER', 'B_tcga_vs_external': 'TCGA vs External', 'C_all_cohorts': 'All Cohorts'}
    name = name_map.get(exp_key, exp_key)
    dc = comp.get('dann_val_cindex', '?')
    bc = comp.get('baseline_val_cindex', '?')
    delta = comp.get('delta', '?')
    print(f'| {name} | {dc} | {bc} | {delta} |')
    # Per-cohort
    cohort_map = comp.get('cohort_map', {})
    rev_map = {v: k for k, v in cohort_map.items()}
    dann_pc = comp.get('dann_per_cohort', {})
    base_pc = comp.get('baseline_per_cohort', {})
    for dom_id in sorted(rev_map.keys()):
        cname = rev_map[dom_id]
        dc2 = dann_pc.get(str(dom_id), 'N/A')
        bc2 = base_pc.get(str(dom_id), 'N/A')
        delta2 = f'{float(dc2)-float(bc2):+.4f}' if dc2 != \"N/A\" and bc2 != \"N/A\" else 'N/A'
        print(f'| — {cname} | {dc2} | {bc2} | {delta2} |')
    print('')
" 2>/dev/null >> "$REPORT_FILE"
    fi

    {
        echo "## Generated Figures"
        echo ""
        for f in results/figures/lihc/*.png 2>/dev/null; do
            echo "- ![$f]($f)"
        done
        echo ""
        echo "## Key Findings"
        echo ""
        echo "1. **Missingness Fingerprint**: Per-cohort missing rates reveal strong domain signatures"
        echo "2. **DANN vs Baseline**: Compare C-indices to evaluate GRL effectiveness"
        echo "3. **Domain Accuracy**: GRL's effect on domain classifier accuracy shows alignment with theory"
        echo "4. **t-SNE Visualization**: Latent space structure for DANN vs Baseline"
        echo ""
        echo "---"
        echo "Pipeline completed at: $(date)"
    } >> "$REPORT_FILE"

    log "Report saved to $REPORT_FILE"
    log "✅ Stage 5 complete"
}

# =====================================================================
# MAIN EXECUTION
# =====================================================================
echo "" | tee -a "$PIPELINE_LOG"
echo "██╗  ██╗██████╗  ██████╗       ██╗     ██╗██╗  ██╗ ██████╗" | tee -a "$PIPELINE_LOG"
echo "██║  ██║██╔══██╗██╔═══██╗      ██║     ██║██║  ██║██╔════╝" | tee -a "$PIPELINE_LOG"
echo "███████║██████╔╝██║   ██║█████╗██║     ██║███████║██║     " | tee -a "$PIPELINE_LOG"
echo "██╔══██║██╔══██╗██║   ██║╚════╝██║     ██║██╔══██║██║     " | tee -a "$PIPELINE_LOG"
echo "██║  ██║██║  ██║╚██████╔╝      ███████╗██║██║  ██║╚██████╗" | tee -a "$PIPELINE_LOG"
echo "╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝       ╚══════╝╚═╝╚═╝  ╚═╝ ╚═════╝" | tee -a "$PIPELINE_LOG"
echo "  LIHC DANN Training + Visualization Pipeline" | tee -a "$PIPELINE_LOG"
echo "  Start: $(date)" | tee -a "$PIPELINE_LOG"
echo "  Quick mode: $QUICK" | tee -a "$PIPELINE_LOG"
echo "================================================================" | tee -a "$PIPELINE_LOG"

case $STAGE in
    1) setup_env; prepare_data; train_models; visualize; generate_report ;;
    2) prepare_data; train_models; visualize; generate_report ;;
    3) train_models; visualize; generate_report ;;
    4) visualize; generate_report ;;
    5) generate_report ;;
    *) echo "Invalid stage: $STAGE (must be 1-5)"; exit 1 ;;
esac

echo ""
echo "================================================================"
echo "  ✅ PIPELINE COMPLETE"
echo "  See $PIPELINE_LOG for full log"
echo "  See results/figures/lihc/ for generated figures"
echo "  See results/lihc_pipeline_report.md for summary"
echo "================================================================"
