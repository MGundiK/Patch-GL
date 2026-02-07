# xpatch_gl_colab.py
"""
Google Colab Cells for xPatch-GL Testing

Copy each cell block to Colab to test the architecture.
"""

# =============================================================================
# CELL 1: Setup and Imports
# =============================================================================
"""
# Install dependencies (if needed)
# !pip install torch numpy pandas matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import math
import time

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
"""

# =============================================================================
# CELL 2: RevIN Module
# =============================================================================
"""
class RevIN(nn.Module):
    '''Reversible Instance Normalization'''
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
            
    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        if mode == 'norm':
            self._mean = x.mean(dim=1, keepdim=True)
            self._std = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / self.affine_weight
            x = x * self._std + self._mean
            return x

print("✓ RevIN defined")
"""

# =============================================================================
# CELL 3: RoRA Module
# =============================================================================
"""
class RoRA(nn.Module):
    '''Rotational Rank Adaptation'''
    
    def __init__(self, dim: int, rank: int = 4, scale_init: float = 0.01):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.U = nn.Parameter(torch.randn(dim, rank) * scale_init)
        self.V = nn.Parameter(torch.randn(dim, rank) * scale_init)
        self.gate = nn.Parameter(torch.tensor(-2.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Build skew-symmetric: Ω = UV^T - VU^T
        UV = self.U @ self.V.T
        omega = UV - UV.T
        
        # Gate and scale
        gate = torch.sigmoid(self.gate)
        omega = omega * gate
        
        # Cayley transform: R = (I + Ω/2)(I - Ω/2)^{-1}
        I = torch.eye(self.dim, device=x.device, dtype=x.dtype)
        half_omega = omega / 2
        R = torch.linalg.solve(I - half_omega, I + half_omega)
        
        return x @ R.T


class PatchRoRA(nn.Module):
    '''RoRA for patch-based models'''
    
    def __init__(self, d_model: int, num_patches: int, rank: int = 4, mode: str = 'feature'):
        super().__init__()
        self.mode = mode
        
        if mode in ['feature', 'both']:
            self.rora_feature = RoRA(d_model, rank=rank)
            self.norm_feature = nn.LayerNorm(d_model)
        if mode in ['patch', 'both']:
            self.rora_patch = RoRA(num_patches, rank=rank)
            self.norm_patch = nn.LayerNorm(num_patches)
            
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        if self.mode == 'feature':
            x = self.rora_feature(self.norm_feature(x))
        elif self.mode == 'patch':
            x = x.transpose(-1, -2)
            x = self.rora_patch(self.norm_patch(x))
            x = x.transpose(-1, -2)
        elif self.mode == 'both':
            x = self.rora_feature(self.norm_feature(x))
            x = x.transpose(-1, -2)
            x = self.rora_patch(self.norm_patch(x))
            x = x.transpose(-1, -2)
            
        return residual + self.scale * x

print("✓ RoRA modules defined")
"""

# =============================================================================
# CELL 4: Intra-Patch and Inter-Patch Modules
# =============================================================================
"""
class IntraPatchModule(nn.Module):
    '''Local processing within each patch'''
    
    def __init__(self, patch_len: int, num_patches: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = patch_len * expansion
        
        self.fc1 = nn.Linear(patch_len, hidden)
        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(num_patches)
        
        self.conv = nn.Conv1d(num_patches, num_patches, 3, padding=1, groups=num_patches)
        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(num_patches)
        
        self.fc2 = nn.Linear(hidden, patch_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.bn1(self.act1(self.fc1(x)))
        x = self.bn2(self.act2(self.conv(x)))
        x = self.dropout(self.fc2(x))
        return x + residual


class InterPatchModule(nn.Module):
    '''Global processing across patches'''
    
    def __init__(self, patch_len: int, num_patches: int, hidden_factor: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = num_patches * hidden_factor
        
        self.mlp = nn.Sequential(
            nn.Linear(num_patches, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_patches),
            nn.Dropout(dropout),
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        # Pool across patch dimension
        pooled = x.mean(dim=-1)  # [BC, N]
        weights = torch.sigmoid(self.mlp(pooled))
        weights = weights.unsqueeze(-1)
        return x * (1 + self.scale * weights)

print("✓ Intra/Inter-Patch modules defined")
"""

# =============================================================================
# CELL 5: GLBlock
# =============================================================================
"""
class GLBlock(nn.Module):
    '''Global-Local Block: Core component of xPatch-GL'''
    
    def __init__(
        self, 
        patch_len: int, 
        num_patches: int, 
        expansion: int = 4,
        fusion_type: str = 'gated',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        
        self.intra = IntraPatchModule(patch_len, num_patches, expansion, dropout)
        self.inter = InterPatchModule(patch_len, num_patches, 2, dropout)
        
        if fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(patch_len * 2, patch_len),
                nn.Sigmoid(),
            )
        elif fusion_type == 'concat':
            self.proj = nn.Linear(patch_len * 2, patch_len)
            
        self.norm = nn.LayerNorm(patch_len)
        
    def forward(self, x):
        local_out = self.intra(x)
        global_out = self.inter(x)
        
        if self.fusion_type == 'gated':
            concat = torch.cat([local_out, global_out], dim=-1)
            gate = self.gate(concat)
            fused = gate * local_out + (1 - gate) * global_out
        elif self.fusion_type == 'add':
            fused = local_out + global_out
        elif self.fusion_type == 'concat':
            fused = self.proj(torch.cat([local_out, global_out], dim=-1))
            
        return self.norm(x + fused)

print("✓ GLBlock defined")
"""

# =============================================================================
# CELL 6: Complete xPatch-GL Network
# =============================================================================
"""
class NetworkGL(nn.Module):
    '''xPatch-GL: Global-Local Patch Network'''
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int = 16,
        stride: int = 8,
        num_gl_blocks: int = 2,
        expansion: int = 4,
        fusion_type: str = 'gated',
        use_rora: bool = False,
        rora_rank: int = 4,
        ema_alpha: float = 0.2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.stride = stride
        self.ema_alpha = ema_alpha
        
        # Number of patches
        self.padding = nn.ReplicationPad1d((0, stride))
        self.num_patches = (seq_len - patch_len) // stride + 2
        
        # RevIN
        self.revin = RevIN(enc_in)
        
        # GLBlocks
        self.gl_blocks = nn.ModuleList([
            GLBlock(patch_len, self.num_patches, expansion, fusion_type, dropout)
            for _ in range(num_gl_blocks)
        ])
        
        # RoRA (optional)
        self.use_rora = use_rora
        if use_rora:
            self.rora = PatchRoRA(patch_len, self.num_patches, rora_rank)
        
        # Seasonal head
        self.seasonal_head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.num_patches * patch_len, pred_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len * 2, pred_len),
        )
        
        # Trend head
        ds_factor = max(1, seq_len // 64)
        # Correct calculation: length after t[:, ::ds_factor] slicing
        ds_len = (seq_len - 1) // ds_factor + 1
        self.ds_factor = ds_factor
        self.trend_head = nn.Sequential(
            nn.Linear(ds_len, 256),
            nn.GELU(),
            nn.Linear(256, pred_len),
        )
        
        # Fusion
        self.fusion = nn.Linear(pred_len * 2, pred_len)
        
    def forward(self, x):
        B, L, C = x.shape
        
        # RevIN
        x = self.revin(x, 'norm')
        
        # EMA decomposition
        trend = torch.zeros_like(x)
        trend[:, 0] = x[:, 0]
        for t in range(1, L):
            trend[:, t] = self.ema_alpha * x[:, t] + (1 - self.ema_alpha) * trend[:, t-1]
        seasonal = x - trend
        
        # Seasonal stream
        s = seasonal.permute(0, 2, 1).reshape(B * C, L)
        s = self.padding(s)
        s = s.unfold(-1, self.patch_len, self.stride)  # [BC, N, P]
        
        for block in self.gl_blocks:
            s = block(s)
            
        if self.use_rora:
            s = self.rora(s)
            
        s_pred = self.seasonal_head(s)
        
        # Trend stream
        t = trend.permute(0, 2, 1).reshape(B * C, L)
        if self.ds_factor > 1:
            t = t[:, ::self.ds_factor]
        t_pred = self.trend_head(t)
        
        # Fuse
        out = self.fusion(torch.cat([s_pred, t_pred], dim=-1))
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)
        out = self.revin(out, 'denorm')
        
        return out

print("✓ NetworkGL defined")
"""

# =============================================================================
# CELL 7: Test the Architecture
# =============================================================================
"""
# Test configuration
B, L, C = 32, 336, 7
pred_len = 96

# Create model
model = NetworkGL(
    seq_len=L,
    pred_len=pred_len,
    enc_in=C,
    patch_len=16,
    stride=8,
    num_gl_blocks=2,
    fusion_type='gated',
    use_rora=False,
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"xPatch-GL Configuration:")
print(f"  seq_len={L}, pred_len={pred_len}, enc_in={C}")
print(f"  num_patches={model.num_patches}")
print(f"  Total parameters: {total_params:,}")
print()

# Forward pass test
x = torch.randn(B, L, C).to(device)
with torch.no_grad():
    y = model(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Expected:     [{B}, {pred_len}, {C}]")
assert y.shape == (B, pred_len, C), "Shape mismatch!"
print("✓ Forward pass successful!")
"""

# =============================================================================
# CELL 8: Test with RoRA
# =============================================================================
"""
# Model with RoRA
model_rora = NetworkGL(
    seq_len=L,
    pred_len=pred_len,
    enc_in=C,
    patch_len=16,
    stride=8,
    num_gl_blocks=2,
    fusion_type='gated',
    use_rora=True,
    rora_rank=4,
).to(device)

rora_params = sum(p.numel() for p in model_rora.parameters())
print(f"xPatch-GL + RoRA:")
print(f"  Parameters: {rora_params:,}")
print(f"  Overhead:   {rora_params - total_params:,} (+{100*(rora_params-total_params)/total_params:.1f}%)")

with torch.no_grad():
    y_rora = model_rora(x)
print(f"  Output shape: {y_rora.shape}")
print("✓ RoRA variant works!")
"""

# =============================================================================
# CELL 9: Speed Benchmark
# =============================================================================
"""
import time

# Warmup
for _ in range(10):
    _ = model(x)
if torch.cuda.is_available():
    torch.cuda.synchronize()

# Benchmark
n_runs = 100
start = time.time()
for _ in range(n_runs):
    _ = model(x)
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Speed Benchmark (xPatch-GL):")
print(f"  {n_runs} forward passes in {elapsed:.2f}s")
print(f"  {elapsed/n_runs*1000:.2f}ms per forward pass")
print(f"  {B * n_runs / elapsed:.0f} samples/sec")
"""

# =============================================================================
# CELL 10: Compare Variants
# =============================================================================
"""
# Compare different configurations
configs = [
    ('GL-Gated', dict(fusion_type='gated', num_gl_blocks=2, use_rora=False)),
    ('GL-Add', dict(fusion_type='add', num_gl_blocks=2, use_rora=False)),
    ('GL-RoRA', dict(fusion_type='gated', num_gl_blocks=2, use_rora=True)),
    ('GL-Deep', dict(fusion_type='gated', num_gl_blocks=3, use_rora=False)),
]

print("Model Comparison:")
print("-" * 50)
print(f"{'Name':<12} {'Params':>12} {'Time (ms)':>12}")
print("-" * 50)

for name, cfg in configs:
    model = NetworkGL(seq_len=L, pred_len=pred_len, enc_in=C, **cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    
    # Time it
    for _ in range(5):
        _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(20):
        _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.time() - start) / 20 * 1000
    
    print(f"{name:<12} {params:>12,} {ms:>12.2f}")

print("-" * 50)
"""

# =============================================================================
# CELL 11: Visualize GLBlock Activations
# =============================================================================
"""
# Hook to capture intermediate activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu()
    return hook

# Register hooks on first GLBlock
model.gl_blocks[0].intra.register_forward_hook(get_activation('intra'))
model.gl_blocks[0].inter.register_forward_hook(get_activation('inter'))

# Forward pass
with torch.no_grad():
    _ = model(x[:4])  # Just 4 samples

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].imshow(activations['intra'][0].numpy(), aspect='auto', cmap='coolwarm')
axes[0].set_title('Intra-Patch (Local) Output')
axes[0].set_xlabel('Patch Features')
axes[0].set_ylabel('Patches')

axes[1].imshow(activations['inter'][0].numpy(), aspect='auto', cmap='coolwarm')
axes[1].set_title('Inter-Patch (Global) Output')
axes[1].set_xlabel('Patch Features')
axes[1].set_ylabel('Patches')

plt.tight_layout()
plt.savefig('gl_activations.png', dpi=150)
plt.show()
print("✓ Saved gl_activations.png")
"""

# =============================================================================
# CELL 12: Quick Training Test
# =============================================================================
"""
# Simple training loop on synthetic data
print("Quick training test on synthetic data...")

# Synthetic data: sinusoid with noise
def generate_data(n_samples, seq_len, pred_len, channels):
    t = torch.linspace(0, 4*np.pi, seq_len + pred_len).unsqueeze(0).unsqueeze(-1)
    x = torch.sin(t) + 0.1 * torch.randn(n_samples, seq_len + pred_len, channels)
    return x[:, :seq_len], x[:, seq_len:]

# Generate data
X_train, Y_train = generate_data(1000, L, pred_len, C)
X_val, Y_val = generate_data(200, L, pred_len, C)

X_train, Y_train = X_train.to(device), Y_train.to(device)
X_val, Y_val = X_val.to(device), Y_val.to(device)

# Model
model = NetworkGL(seq_len=L, pred_len=pred_len, enc_in=C, num_gl_blocks=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train
train_losses = []
val_losses = []

for epoch in range(20):
    # Train
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, Y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    # Validate
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, Y_val)
        val_losses.append(val_loss.item())
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")

# Plot
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('xPatch-GL Training on Synthetic Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('gl_training.png', dpi=150)
plt.show()
print("✓ Saved gl_training.png")
"""

# =============================================================================
# CELL 13: Export for Full Training
# =============================================================================
"""
print("\\n" + "="*60)
print("xPatch-GL Architecture Summary")
print("="*60)
print('''
Key Components:
1. RevIN: Handles distribution shift
2. EMA Decomposition: Separates trend from seasonal
3. GLBlocks: Explicit local-global separation
   - IntraPatchModule: CNN within each patch (local)
   - InterPatchModule: MLP across patches (global)
   - Gated fusion: Learns to balance local vs global
4. RoRA (optional): Geometric alignment via rotation
5. Dual heads: Seasonal (from GLBlocks) + Trend (MLP)
6. Fusion: Combines seasonal and trend predictions

Innovations over xPatch:
- EXPLICIT separation of local/global (from GLCN paper)
- Gated fusion between local and global streams
- Optional RoRA for geometric alignment
- Cleaner decomposition pathway

To run full experiments:
1. Copy the network code to your codebase
2. Update run.py to support xpatch_gl model
3. Run experiments with runner_gl.sh
''')
"""

print("xPatch-GL Colab cells ready!")
print("Copy cells to Google Colab to test the architecture.")
