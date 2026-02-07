#!/bin/bash
# runner_gl.sh - Experiment runner for xPatch-GL
#
# Usage: bash runner_gl.sh
#
# Experiments:
# 1. gl_gated: xPatch-GL with gated fusion (default)
# 2. gl_add: xPatch-GL with additive fusion
# 3. gl_rora: xPatch-GL with RoRA enabled
# 4. gl_lite: Lightweight variant
# 5. gl_deep: 3 GLBlocks instead of 2

set -e

# Common settings
EPOCHS=30
BATCH=32
LR=0.001
PATIENCE=5
SEQ_LEN=336

echo "=========================================="
echo "xPatch-GL Experiments"
echo "=========================================="
echo "Using run_gl.py for all experiments"
echo ""

# Create results directory
mkdir -p results_gl

# ============================================================
# Helper function for running experiments
# ============================================================
run_experiment() {
    local exp_name=$1
    local dataset=$2
    local data_type=$3
    local data_file=$4
    local enc_in=$5
    local pred_len=$6
    local extra_args=$7
    
    echo "  → ${exp_name}: ${dataset} pred_len=${pred_len}"
    
    python run_gl.py \
        --data ${data_type} \
        --data_path ${data_file} \
        --enc_in ${enc_in} \
        --seq_len ${SEQ_LEN} \
        --pred_len ${pred_len} \
        --patch_len 16 \
        --stride 8 \
        --ema_alpha 0.2 \
        --train_epochs ${EPOCHS} \
        --batch_size ${BATCH} \
        --learning_rate ${LR} \
        --patience ${PATIENCE} \
        --itr 1 \
        ${extra_args} \
        2>&1 | tee -a results_gl/${exp_name}_${dataset}_${pred_len}.log
}

# ============================================================
# Dataset configurations
# Format: dataset_name data_type data_file enc_in
# ============================================================
declare -A DATASETS
DATASETS["ETTh1"]="ETTh1 ETTh1.csv 7"
DATASETS["ETTh2"]="ETTh2 ETTh2.csv 7"
DATASETS["ETTm1"]="ETTm1 ETTm1.csv 7"
DATASETS["ETTm2"]="ETTm2 ETTm2.csv 7"
DATASETS["weather"]="custom weather.csv 21"
DATASETS["exchange"]="custom exchange_rate.csv 8"

PRED_LENS="96 192 336 720"

# ============================================================
# Experiment 1: GL Gated (default)
# ============================================================
echo "[1/5] GL Gated Fusion"
for dataset in ETTh1 ETTm1 weather exchange; do
    IFS=' ' read -r data_type data_file enc_in <<< "${DATASETS[$dataset]}"
    for pred_len in $PRED_LENS; do
        run_experiment "gl_gated" "$dataset" "$data_type" "$data_file" "$enc_in" "$pred_len" \
            "--num_gl_blocks 2 --fusion_type gated --use_rora 0"
    done
done

# ============================================================
# Experiment 2: GL Add (simpler fusion)
# ============================================================
echo ""
echo "[2/5] GL Add Fusion"
for dataset in ETTh1 ETTm1 weather exchange; do
    IFS=' ' read -r data_type data_file enc_in <<< "${DATASETS[$dataset]}"
    for pred_len in $PRED_LENS; do
        run_experiment "gl_add" "$dataset" "$data_type" "$data_file" "$enc_in" "$pred_len" \
            "--num_gl_blocks 2 --fusion_type add --use_rora 0"
    done
done

# ============================================================
# Experiment 3: GL + RoRA
# ============================================================
echo ""
echo "[3/5] GL + RoRA"
for dataset in ETTh1 ETTm1 weather exchange; do
    IFS=' ' read -r data_type data_file enc_in <<< "${DATASETS[$dataset]}"
    for pred_len in $PRED_LENS; do
        run_experiment "gl_rora" "$dataset" "$data_type" "$data_file" "$enc_in" "$pred_len" \
            "--num_gl_blocks 2 --fusion_type gated --use_rora 1 --rora_rank 4"
    done
done

# ============================================================
# Experiment 4: GL Lite
# ============================================================
echo ""
echo "[4/5] GL Lite"
for dataset in ETTh1 ETTm1 weather exchange; do
    IFS=' ' read -r data_type data_file enc_in <<< "${DATASETS[$dataset]}"
    for pred_len in $PRED_LENS; do
        run_experiment "gl_lite" "$dataset" "$data_type" "$data_file" "$enc_in" "$pred_len" \
            "--gl_variant lite --expansion_factor 2"
    done
done

# ============================================================
# Experiment 5: GL Deep (3 blocks)
# ============================================================
echo ""
echo "[5/5] GL Deep (3 blocks)"
for dataset in ETTh1 ETTm1 weather exchange; do
    IFS=' ' read -r data_type data_file enc_in <<< "${DATASETS[$dataset]}"
    for pred_len in $PRED_LENS; do
        run_experiment "gl_deep" "$dataset" "$data_type" "$data_file" "$enc_in" "$pred_len" \
            "--num_gl_blocks 3 --fusion_type gated --use_rora 0"
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results saved in results_gl/"
echo ""
echo "To parse results:"
echo '  grep -r "mse:" results_gl/ | sort'
