#!/bin/bash
# runner_xstyle.sh - Full experiment suite for xPatch-GL using xPatch hyper-parameters
#
# Usage: bash runner_xstyle.sh
#
# Runs xPatch-GL on all standard benchmarks with patience=10

set -e

# Common settings (matching xPatch defaults)
EPOCHS=100
BATCH=32
LR=0.0001
PATIENCE=10  # Changed from 5 to 10
SEQ_LEN=336

echo "=========================================="
echo "xPatch-GL Full Experiment Suite"
echo "=========================================="
echo "Settings: epochs=$EPOCHS, batch=$BATCH, lr=$LR, patience=$PATIENCE"
echo ""

# Create results directory
mkdir -p results_336_rora

# ============================================================
# GL Gated (default configuration)
# ============================================================

# ETTh1
echo "[ETTh1] Running all horizons..."
for pred_len in 96 192 336 720; do
    echo "  → ETTh1 pred_len=$pred_len"
    python run_gl.py \
        --data ETTh1 \
        --data_path ETTh1.csv \
        --enc_in 7 \
        --seq_len $SEQ_LEN \
        --pred_len $pred_len \
        --patch_len 16 \
        --stride 8 \
        --num_gl_blocks 2 \
        --fusion_type gated \
        --use_rora 1 \
        --rora_rank 4 \
        --ema_alpha 0.3 \
        --train_epochs $EPOCHS \
        --batch_size 2048 \
        --learning_rate 0.0001 \
        --lradj 'sigmoid'\
        --patience $PATIENCE \
        --itr 1 \
        2>&1 | tee results_336_rora/gl_gated_ETTh1_${pred_len}.log
done

# ETTh2
echo ""
echo "[ETTh2] Running all horizons..."
for pred_len in 96 192 336 720; do
    echo "  → ETTh2 pred_len=$pred_len"
    python run_gl.py \
        --data ETTh2 \
        --data_path ETTh2.csv \
        --enc_in 7 \
        --seq_len $SEQ_LEN \
        --pred_len $pred_len \
        --patch_len 16 \
        --stride 8 \
        --num_gl_blocks 2 \
        --fusion_type gated \
        --use_rora 1 \
        --rora_rank 4 \
        --ema_alpha 0.3 \
        --train_epochs $EPOCHS \
        --batch_size 2048 \
        --learning_rate $LR \
        --lradj 'sigmoid'\
        --patience $PATIENCE \
        --itr 1 \
        2>&1 | tee results_336_rora/gl_gated_ETTh2_${pred_len}.log
done

# ETTm1
echo ""
echo "[ETTm1] Running all horizons..."
for pred_len in 96 192 336 720; do
    echo "  → ETTm1 pred_len=$pred_len"
    python run_gl.py \
        --data ETTm1 \
        --data_path ETTm1.csv \
        --enc_in 7 \
        --seq_len $SEQ_LEN \
        --pred_len $pred_len \
        --patch_len 16 \
        --stride 8 \
        --num_gl_blocks 2 \
        --fusion_type gated \
        --use_rora 1 \
        --rora_rank 4 \
        --ema_alpha 0.3 \
        --train_epochs $EPOCHS \
        --batch_size 2048 \
        --learning_rate $LR \
        --lradj 'sigmoid'\
        --patience $PATIENCE \
        --itr 1 \
        2>&1 | tee results_336_rora/gl_gated_ETTm1_${pred_len}.log
done

# ETTm2
echo ""
echo "[ETTm2] Running all horizons..."
for pred_len in 96 192 336 720; do
    echo "  → ETTm2 pred_len=$pred_len"
    python run_gl.py \
        --data ETTm2 \
        --data_path ETTm2.csv \
        --enc_in 7 \
        --seq_len $SEQ_LEN \
        --pred_len $pred_len \
        --patch_len 16 \
        --stride 8 \
        --num_gl_blocks 2 \
        --fusion_type gated \
        --use_rora 1 \
        --rora_rank 4 \
        --ema_alpha 0.3 \
        --train_epochs $EPOCHS \
        --batch_size 2048 \
        --learning_rate $LR \
        --lradj 'sigmoid'\
        --patience $PATIENCE \
        --itr 1 \
        2>&1 | tee results_336_rora/gl_gated_ETTm2_${pred_len}.log
done

# Weather
echo ""
echo "[Weather] Running all horizons..."
for pred_len in 96 192 336 720; do
    echo "  → Weather pred_len=$pred_len"
    python run_gl.py \
        --data custom \
        --data_path weather.csv \
        --enc_in 21 \
        --seq_len $SEQ_LEN \
        --pred_len $pred_len \
        --patch_len 16 \
        --stride 8 \
        --num_gl_blocks 2 \
        --fusion_type gated \
        --use_rora 1 \
        --rora_rank 4 \
        --ema_alpha 0.3 \
        --train_epochs $EPOCHS \
        --batch_size 1024 \
        --learning_rate $LR \
        --lradj 'sigmoid'\
        --patience $PATIENCE \
        --itr 1 \
        2>&1 | tee results_336_rora/gl_gated_weather_${pred_len}.log
done

# Exchange
echo ""
echo "[Exchange] Running all horizons..."
for pred_len in 96 192 336 720; do
    echo "  → Exchange pred_len=$pred_len"
    python run_gl.py \
        --data custom \
        --data_path exchange_rate.csv \
        --enc_in 8 \
        --seq_len 96  \
        --pred_len $pred_len \
        --patch_len 16 \
        --stride 8 \
        --num_gl_blocks 2 \
        --fusion_type gated \
        --use_rora 1 \
        --rora_rank 4 \
        --ema_alpha 0.3 \
        --train_epochs $EPOCHS \
        --batch_size $BATCH \
        --learning_rate $LR \
        --lradj 'sigmoid'\
        --patience $PATIENCE \
        --itr 1 \
        2>&1 | tee results_336_rora/gl_gated_exchange_${pred_len}.log
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results summary:"
echo "----------------"
grep -h "^mse:" results_336_rora/*.log | head -30
echo ""
echo "Full results in results_336_rora/"
echo "Parse with: grep 'Final Results\|MSE:' results_336_rora/*.log"
