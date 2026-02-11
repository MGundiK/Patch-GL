#!/bin/bash
# =============================================================================
# xPatch-GL Hybrid: Experiment Runner
# =============================================================================

# Settings
EPOCHS=30
LR=0.0001
PATIENCE=10
SEQ_LEN=336
PATCH_LEN=16
STRIDE=8

mkdir -p results_hybrid

# Function to run single experiment with EXPLICIT naming
run_exp() {
    local name=$1       # e.g., ETTh1, weather, exchange_rate
    local data=$2       # e.g., ETTh1, custom
    local data_path=$3  # e.g., ETTh1.csv, weather.csv
    local enc_in=$4
    local pred_len=$5
    local variant=$6
    local batch=$7
    
    echo "============================================================"
    echo "Running: ${name} pred_len=${pred_len} variant=${variant}"
    echo "============================================================"
    
    python run_hybrid.py \
        --is_training 1 \
        --data ${data} \
        --data_path ${data_path} \
        --root_path ./dataset/ \
        --enc_in ${enc_in} \
        --seq_len ${SEQ_LEN} \
        --pred_len ${pred_len} \
        --variant ${variant} \
        --train_epochs ${EPOCHS} \
        --batch_size ${batch} \
        --learning_rate ${LR} \
        --patience ${PATIENCE} \
        2>&1 | tee "results_hybrid/hybrid_${variant}_${name}_${pred_len}.log"
}

# Run all horizons for a dataset
run_all_horizons() {
    local name=$1
    local data=$2
    local data_path=$3
    local enc_in=$4
    local variant=$5
    local batch=$6
    
    for pl in 96 192 336 720; do
        run_exp ${name} ${data} ${data_path} ${enc_in} ${pl} ${variant} ${batch}
    done
}

# ==== DATASET DEFINITIONS (explicit names) ====
run_ETTh1()   { run_all_horizons "ETTh1"   "ETTh1"  "ETTh1.csv"          7  "$1" 2048; }
run_ETTh2()   { run_all_horizons "ETTh2"   "ETTh2"  "ETTh2.csv"          7  "$1" 2048; }
run_ETTm1()   { run_all_horizons "ETTm1"   "ETTm1"  "ETTm1.csv"          7  "$1" 2048; }
run_ETTm2()   { run_all_horizons "ETTm2"   "ETTm2"  "ETTm2.csv"          7  "$1" 2048; }
run_weather() { run_all_horizons "weather" "custom" "weather.csv"        21 "$1" 1024; }
run_exchange(){ run_all_horizons "exchange_rate" "custom" "exchange_rate.csv" 8 "$1" 32; }

# Run all datasets
run_all() {
    local variant=$1
    run_ETTh1 ${variant}
    run_ETTh2 ${variant}
    run_ETTm1 ${variant}
    run_ETTm2 ${variant}
    run_weather ${variant}
    run_exchange ${variant}
}

# Ablation study
run_ablation() {
    local dataset=$1
    for v in full no_agg no_ms no_gate baseline; do
        echo ">>> Ablation variant: ${v}"
        case ${dataset} in
            ETTh1)   run_ETTh1 ${v} ;;
            ETTh2)   run_ETTh2 ${v} ;;
            ETTm1)   run_ETTm1 ${v} ;;
            ETTm2)   run_ETTm2 ${v} ;;
            weather) run_weather ${v} ;;
            exchange) run_exchange ${v} ;;
        esac
    done
}

# ==== MAIN ====
VARIANT=${1:-full}
DATASET=${2:-all}

echo "============================================================"
echo "xPatch-GL Hybrid Runner"
echo "Variant: ${VARIANT}, Dataset: ${DATASET}"
echo "============================================================"

if [ "${VARIANT}" = "ablation" ]; then
    run_ablation ${DATASET}
elif [ "${DATASET}" = "all" ]; then
    run_all ${VARIANT}
else
    case ${DATASET} in
        ETTh1)   run_ETTh1 ${VARIANT} ;;
        ETTh2)   run_ETTh2 ${VARIANT} ;;
        ETTm1)   run_ETTm1 ${VARIANT} ;;
        ETTm2)   run_ETTm2 ${VARIANT} ;;
        weather) run_weather ${VARIANT} ;;
        exchange) run_exchange ${VARIANT} ;;
        *) echo "Unknown dataset: ${DATASET}" ;;
    esac
fi

echo "============================================================"
echo "Done! Results in results_hybrid/"
echo "============================================================"
