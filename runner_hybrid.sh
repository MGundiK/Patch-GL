#!/bin/bash
# =============================================================================
# xPatch-GL Hybrid: Experiment Runner
# =============================================================================
#
# Usage:
#   bash runner_hybrid.sh [variant] [dataset]
#
# Variants:
#   full     - All GLCN enhancements (aggregate + multiscale + gating)
#   no_agg   - No aggregate conv
#   no_ms    - No multi-scale conv
#   no_gate  - No gating
#   baseline - Pure xPatch style (no enhancements)
#   ablation - Run all variants for ablation study
#   all      - Run full variant on all datasets
#
# Datasets:
#   ETTh1, ETTh2, ETTm1, ETTm2, weather, exchange
#
# Examples:
#   bash runner_hybrid.sh full ETTh1
#   bash runner_hybrid.sh ablation ETTh1
#   bash runner_hybrid.sh all
# =============================================================================

# Settings (matching run_gl.py defaults)
EPOCHS=30
BATCH=32
LR=0.0001
PATIENCE=5
SEQ_LEN=336
EMA_ALPHA=0.3
PATCH_LEN=16
STRIDE=8

# Create results directory
mkdir -p results_hybrid

# Function to run a single experiment
run_experiment() {
    local dataset=$1
    local data_path=$2
    local enc_in=$3
    local pred_len=$4
    local variant=$5
    
    echo "============================================================"
    echo "Running: $dataset pred_len=$pred_len variant=$variant"
    echo "============================================================"
    
    python run_hybrid.py \
        --is_training 1 \
        --data $dataset \
        --data_path $data_path \
        --root_path ./dataset/ \
        --enc_in $enc_in \
        --seq_len $SEQ_LEN \
        --pred_len $pred_len \
        --patch_len $PATCH_LEN \
        --stride $STRIDE \
        --ema_alpha $EMA_ALPHA \
        --variant $variant \
        --train_epochs $EPOCHS \
        --batch_size $BATCH \
        --learning_rate $LR \
        --patience $PATIENCE \
        2>&1 | tee results_hybrid/hybrid_${variant}_${dataset}_${pred_len}.log
}

# Function to run all prediction lengths for a dataset
run_all_pred_lens() {
    local dataset=$1
    local data_path=$2
    local enc_in=$3
    local variant=$4
    
    for pred_len in 96 192 336 720; do
        run_experiment $dataset $data_path $enc_in $pred_len $variant
    done
}

# Function to run all datasets
run_all_datasets() {
    local variant=$1
    
    # ETT datasets (7 features)
    run_all_pred_lens "ETTh1" "ETTh1.csv" 7 $variant
    run_all_pred_lens "ETTh2" "ETTh2.csv" 7 $variant
    run_all_pred_lens "ETTm1" "ETTm1.csv" 7 $variant
    run_all_pred_lens "ETTm2" "ETTm2.csv" 7 $variant
    
    # Weather (21 features)
    run_all_pred_lens "custom" "weather.csv" 21 $variant
    
    # Exchange (8 features)  
    run_all_pred_lens "custom" "exchange_rate.csv" 8 $variant
}

# Function to run ablation study on one dataset
run_ablation() {
    local dataset=$1
    local data_path=$2
    local enc_in=$3
    
    echo "============================================================"
    echo "ABLATION STUDY: $dataset"
    echo "============================================================"
    
    for variant in full no_agg no_ms no_gate baseline; do
        echo ""
        echo ">>> Variant: $variant"
        run_all_pred_lens $dataset $data_path $enc_in $variant
    done
}

# Parse arguments
VARIANT=${1:-full}
DATASET=${2:-all}

echo "============================================================"
echo "xPatch-GL Hybrid Experiment Runner"
echo "============================================================"
echo "Variant: $VARIANT"
echo "Dataset: $DATASET"
echo "Settings: EPOCHS=$EPOCHS, BATCH=$BATCH, LR=$LR, PATIENCE=$PATIENCE"
echo "          SEQ_LEN=$SEQ_LEN, PATCH_LEN=$PATCH_LEN, STRIDE=$STRIDE"
echo "============================================================"

# Main logic
if [ "$VARIANT" = "ablation" ]; then
    # Ablation study on specified dataset
    case $DATASET in
        ETTh1) run_ablation "ETTh1" "ETTh1.csv" 7 ;;
        ETTh2) run_ablation "ETTh2" "ETTh2.csv" 7 ;;
        ETTm1) run_ablation "ETTm1" "ETTm1.csv" 7 ;;
        ETTm2) run_ablation "ETTm2" "ETTm2.csv" 7 ;;
        weather) run_ablation "custom" "weather.csv" 21 ;;
        exchange) run_ablation "custom" "exchange_rate.csv" 8 ;;
        all) 
            run_ablation "ETTh1" "ETTh1.csv" 7
            run_ablation "ETTh2" "ETTh2.csv" 7
            ;;
        *) echo "Unknown dataset: $DATASET" ;;
    esac
elif [ "$VARIANT" = "all" ]; then
    # Run full variant on all datasets
    run_all_datasets "full"
else
    # Run specific variant on specific dataset
    case $DATASET in
        ETTh1) run_all_pred_lens "ETTh1" "ETTh1.csv" 7 $VARIANT ;;
        ETTh2) run_all_pred_lens "ETTh2" "ETTh2.csv" 7 $VARIANT ;;
        ETTm1) run_all_pred_lens "ETTm1" "ETTm1.csv" 7 $VARIANT ;;
        ETTm2) run_all_pred_lens "ETTm2" "ETTm2.csv" 7 $VARIANT ;;
        weather) run_all_pred_lens "custom" "weather.csv" 21 $VARIANT ;;
        exchange) run_all_pred_lens "custom" "exchange_rate.csv" 8 $VARIANT ;;
        all) run_all_datasets $VARIANT ;;
        *) echo "Unknown dataset: $DATASET" ;;
    esac
fi

echo ""
echo "============================================================"
echo "All experiments completed!"
echo "Results saved to results_hybrid/"
echo "============================================================"
