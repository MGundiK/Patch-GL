# xpatch_gl_standalone.py
"""
Self-contained xPatch-GL for testing in Google Colab or standalone environments.
No external dependencies except PyTorch.

Copy this entire file to Colab and run all cells.
"""

# =============================================================================
# CELL 1: Imports
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# CELL 2: RevIN
# =============================================================================
class RevIN(nn.Module):
    """Reversible Instance Normalization"""
    
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


# =============================================================================
# CELL 3: RoRA Modules
# =============================================================================
class RoRA(nn.Module):
    """Rotational Rank Adaptation via low-rank skew-symmetric generator"""
    
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
        gate = torch.sigmoid(self.gate)
        omega = omega * gate
        
        # Cayley transform: R = (I + Ω/2)(I - Ω/2)^{-1}
        I = torch.eye(self.dim, device=x.device, dtype=x.dtype)
        half_omega = omega / 2
        R = torch.linalg.solve(I - half_omega, I + half_omega)
        return x @ R.T


class PatchRoRA(nn.Module):
    """RoRA adapted for patch-based models"""
    
    def __init__(self, d_model: int, num_patches: int, rank: int = 4, mode: str = 'feature'):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.num_patches = num_patches
        
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


# =============================================================================
# CELL 4: Intra-Patch Module (Local)
# =============================================================================
class IntraPatchModule(nn.Module):
    """Local processing within each patch using depthwise conv"""
    
    def __init__(self, patch_len: int, num_patches: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = patch_len * expansion
        
        self.fc1 = nn.Linear(patch_len, hidden)
        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(num_patches)
        
        # Depthwise conv per patch
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


# =============================================================================
# CELL 5: Inter-Patch Module (Global)
# =============================================================================
class InterPatchModule(nn.Module):
    """Global processing across patches using MLP on pooled features"""
    
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


# =============================================================================
# CELL 6: GLBlock (Core Innovation)
# =============================================================================
class GLBlock(nn.Module):
    """
    Global-Local Block: The core component of xPatch-GL
    
    Combines local (intra-patch) and global (inter-patch) processing
    with learnable fusion.
    """
    
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
        
        # Local stream
        self.intra = IntraPatchModule(patch_len, num_patches, expansion, dropout)
        
        # Global stream
        self.inter = InterPatchModule(patch_len, num_patches, 2, dropout)
        
        # Fusion
        if fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(patch_len * 2, patch_len),
                nn.Sigmoid(),
            )
        elif fusion_type == 'concat':
            self.proj = nn.Linear(patch_len * 2, patch_len)
            
        self.norm = nn.LayerNorm(patch_len)
        
    def forward(self, x):
        # Parallel processing
        local_out = self.intra(x)
        global_out = self.inter(x)
        
        # Fusion
        if self.fusion_type == 'gated':
            concat = torch.cat([local_out, global_out], dim=-1)
            gate = self.gate(concat)
            fused = gate * local_out + (1 - gate) * global_out
        elif self.fusion_type == 'add':
            fused = local_out + global_out
        elif self.fusion_type == 'concat':
            fused = self.proj(torch.cat([local_out, global_out], dim=-1))
            
        return self.norm(x + fused)


# =============================================================================
# CELL 7: NetworkGL (Complete Model)
# =============================================================================
class NetworkGL(nn.Module):
    """
    xPatch-GL: Global-Local Patch Network
    
    A lightweight CNN-based architecture that explicitly separates
    local (intra-patch) and global (inter-patch) temporal modeling.
    """
    
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
        
        # Final fusion
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


# =============================================================================
# CELL 8: Test the Model
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("xPatch-GL Architecture Test")
    print("="*60)
    
    # Configuration
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
    print(f"\nModel Configuration:")
    print(f"  seq_len={L}, pred_len={pred_len}, enc_in={C}")
    print(f"  num_patches={model.num_patches}")
    print(f"  Parameters: {total_params:,}")
    
    # Forward pass
    x = torch.randn(B, L, C).to(device)
    with torch.no_grad():
        y = model(x)
    
    print(f"\nForward Pass:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    assert y.shape == (B, pred_len, C), "Shape mismatch!"
    print("  ✓ Shape correct!")
    
    # Test with RoRA
    print(f"\nWith RoRA:")
    model_rora = NetworkGL(
        seq_len=L, pred_len=pred_len, enc_in=C,
        use_rora=True, rora_rank=4
    ).to(device)
    
    rora_params = sum(p.numel() for p in model_rora.parameters())
    print(f"  Parameters: {rora_params:,} (+{rora_params - total_params:,})")
    
    with torch.no_grad():
        y_rora = model_rora(x)
    print(f"  Output: {y_rora.shape}")
    print("  ✓ RoRA variant works!")
    
    # Compare fusion types
    print(f"\nFusion Type Comparison:")
    for fusion in ['gated', 'add', 'concat']:
        m = NetworkGL(seq_len=L, pred_len=pred_len, enc_in=C, fusion_type=fusion).to(device)
        params = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            out = m(x)
        print(f"  {fusion}: {params:,} params, output {out.shape}")
    
    # Quick training test on synthetic data
    print(f"\nQuick Training Test (synthetic data):")
    
    # Generate synthetic sinusoidal data
    t = torch.linspace(0, 8*np.pi, L + pred_len).unsqueeze(0).unsqueeze(-1)
    X = torch.sin(t) + 0.1 * torch.randn(500, L + pred_len, C)
    X_train, Y_train = X[:400, :L].to(device), X[:400, L:].to(device)
    X_val, Y_val = X[400:, :L].to(device), X[400:, L:].to(device)
    
    model = NetworkGL(seq_len=L, pred_len=pred_len, enc_in=C).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, Y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, Y_val)
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}: train={loss.item():.4f}, val={val_loss.item():.4f}")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("""
Next steps:
1. Copy layers/network_gl.py to your codebase
2. Copy exp/exp_xpatch_gl.py for training integration
3. Run: python run_gl.py --data ETTh1 --pred_len 96
""")
