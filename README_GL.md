# xPatch-GL: Global-Local Patch Network

A new architecture for long-term time series forecasting that combines:
- **xPatch**: Proven patching + EMA decomposition
- **GLCN**: Explicit global-local separation (from "From global to local" paper)
- **RoRA**: Rotational Rank Adaptation for geometric alignment

## Architecture Overview

```
Input [B, L, C]
    │
    ▼
┌─────────────────┐
│     RevIN       │  ← Handles distribution shift
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ EMA Decompose   │  ← Trend + Seasonal separation
└─────────────────┘
    │
    ├──────────────────────────────────┐
    │ (Seasonal)                       │ (Trend)
    ▼                                  ▼
┌─────────────────┐              ┌─────────────────┐
│    Patching     │              │   Trend Head    │
└─────────────────┘              └─────────────────┘
    │                                  │
    ▼                                  │
┌─────────────────┐                    │
│    GLBlock      │ ← Local + Global   │
│  ┌───────────┐  │                    │
│  │ Intra-Patch│  │ CNN within patch  │
│  └───────────┘  │                    │
│  ┌───────────┐  │                    │
│  │ Inter-Patch│  │ MLP across patches│
│  └───────────┘  │                    │
│  ┌───────────┐  │                    │
│  │   Fusion  │  │ Gated combination  │
│  └───────────┘  │                    │
└─────────────────┘                    │
    │                                  │
    ▼ (optional)                       │
┌─────────────────┐                    │
│     RoRA        │ ← Geometric align  │
└─────────────────┘                    │
    │                                  │
    ▼                                  │
┌─────────────────┐                    │
│ Seasonal Head   │                    │
└─────────────────┘                    │
    │                                  │
    └──────────────┬───────────────────┘
                   │
                   ▼
           ┌─────────────────┐
           │   Final Fusion  │
           └─────────────────┘
                   │
                   ▼
           ┌─────────────────┐
           │     RevIN       │  ← Denormalize
           └─────────────────┘
                   │
                   ▼
           Output [B, pred_len, C]
```

## Key Innovations

### 1. Explicit Global-Local Separation (from GLCN)

Unlike xPatch which mixes local and global in conv layers, xPatch-GL explicitly separates:

- **Intra-Patch (Local)**: fc → depthwise conv within each patch
- **Inter-Patch (Global)**: GlobalAvgPool → MLP across patches
- **Gated Fusion**: Learns to balance local vs global per-position

### 2. RoRA Integration (optional)

Rotational Rank Adaptation provides geometric alignment of the learned representation:
- Learns orthogonal transformation R = exp(Ω)
- Low-rank parameterization: Ω = UV^T - VU^T
- Preserves energy while reorienting feature space

### 3. Clean Decomposition Pathway

- EMA trend extraction (proven winner from xPatch experiments)
- Separate prediction heads for trend and seasonal
- Final fusion combines both streams

## Files

```
layers/
├── network_gl.py      # Main architecture (NetworkGL, NetworkGL_Lite)
├── rora.py            # RoRA modules (RoRA, PatchRoRA)
models/
├── xpatch_gl.py       # Model wrapper for training integration
runner_gl.sh           # Experiment runner script
xpatch_gl_colab.py     # Colab cells for testing
```

## Usage

### Basic Usage

```python
from layers.network_gl import NetworkGL

model = NetworkGL(
    seq_len=336,
    pred_len=96,
    enc_in=7,
    patch_len=16,
    stride=8,
    num_gl_blocks=2,
    fusion_type='gated',
    use_rora=False,
)

x = torch.randn(32, 336, 7)  # [B, L, C]
y = model(x)                  # [B, 96, 7]
```

### With RoRA

```python
model = NetworkGL(
    seq_len=336,
    pred_len=96,
    enc_in=7,
    use_rora=True,
    rora_rank=4,
    rora_mode='feature',
)
```

### Lite Version (fewer parameters)

```python
from layers.network_gl import NetworkGL_Lite

model = NetworkGL_Lite(
    seq_len=336,
    pred_len=96,
    enc_in=7,
)
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| seq_len | - | Input sequence length |
| pred_len | - | Prediction horizon |
| enc_in | - | Number of input channels |
| patch_len | 16 | Length of each patch |
| stride | 8 | Patch stride |
| d_model | patch_len | Model dimension |
| num_gl_blocks | 2 | Number of GLBlocks |
| expansion_factor | 4 | MLP expansion in intra-patch |
| fusion_type | 'gated' | 'gated', 'add', or 'concat' |
| use_rora | False | Enable RoRA geometric alignment |
| rora_rank | 4 | Rank for RoRA |
| rora_mode | 'feature' | 'feature', 'patch', or 'both' |
| ema_alpha | 0.2 | EMA smoothing factor |
| dropout | 0.1 | Dropout rate |

## Experiments

Run the full experiment suite:

```bash
bash runner_gl.sh
```

This runs:
1. `gl_standard`: Gated fusion (default)
2. `gl_add`: Additive fusion
3. `gl_rora`: With RoRA enabled
4. `gl_lite`: Lightweight variant
5. `gl_deep`: 3 GLBlocks
6. `baseline`: Original xPatch for comparison

## Expected Improvements

Based on the GLCN paper claims:
- 1.6% MSE reduction vs SOTA
- 65-86% faster training
- 94%+ parameter reduction vs Transformers

Our hypothesis: Explicit global-local separation may help where xPatch's mixed conv layers struggle.

## References

1. **xPatch**: Original patching + decomposition architecture
2. **GLCN**: "From global to local: A lightweight CNN approach for long-term time series forecasting" (Computers & Electrical Engineering, 2025)
3. **RoRA-Tab**: Rotational Rank Adaptation (Sudjianto, 2026)
4. **RevIN**: Reversible Instance Normalization (Kim et al., ICLR 2021)

## Integration with Existing Codebase

### File Structure

Place the files in your existing project:

```
your_project/
├── data_provider/
│   ├── data_factory.py    # (existing)
│   └── data_loader.py     # (existing)
├── exp/
│   ├── exp_basic.py       # (existing)
│   ├── exp_main.py        # (existing)
│   └── exp_xpatch_gl.py   # ← NEW
├── layers/
│   ├── network_gl.py      # ← NEW
│   └── rora.py            # ← NEW
├── models/
│   └── xpatch_gl.py       # ← NEW (optional wrapper)
├── run_gl.py              # ← NEW
├── runner_gl.sh           # ← NEW
└── xpatch_gl_standalone.py # ← For testing without data
```

### Quick Start

1. **Test standalone** (no data needed):
   ```bash
   python xpatch_gl_standalone.py
   ```

2. **Run single experiment**:
   ```bash
   python run_gl.py --data ETTh1 --pred_len 96 --fusion_type gated
   ```

3. **Run full experiment suite**:
   ```bash
   bash runner_gl.sh
   ```

### Key Arguments

| Argument | Default | Options |
|----------|---------|---------|
| --fusion_type | gated | gated, add, concat |
| --num_gl_blocks | 2 | 1, 2, 3, ... |
| --use_rora | 0 | 0, 1 |
| --rora_rank | 4 | 2, 4, 8, ... |
| --gl_variant | standard | standard, lite |
| --ema_alpha | 0.2 | 0.1 - 0.5 |

### Dataset Configuration

The runner script handles these datasets:
- ETTh1, ETTh2 (hourly, 7 features)
- ETTm1, ETTm2 (15-min, 7 features)
- weather (21 features)
- exchange (8 features)

Make sure your data is in `./dataset/` directory.
