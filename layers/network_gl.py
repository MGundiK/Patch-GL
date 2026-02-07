# layers/network_gl.py
"""
xPatch-GL: Global-Local xPatch Network

A new architecture combining:
1. xPatch's proven components (EMA decomposition, patching, channel independence)
2. GLCN's explicit global-local separation (inter-patch and intra-patch modules)
3. RoRA for geometric alignment of fused representations

Architecture:
    Input → RevIN → Patching
        ├→ Intra-patch (Local): fc → conv within each patch
        └→ Inter-patch (Global): GlobalAvgPool → MLP across patches
        → Fusion (gated/additive)
        → RoRA (optional geometric alignment)
        → Flatten Head
        + Trend Stream (EMA)
        → Prediction

Key innovations:
- Explicit separation of local (within-patch) and global (cross-patch) processing
- Learnable fusion of the two streams
- RoRA provides geometric reorientation without adding complexity
- Maintains channel independence for robustness

Reference:
- xPatch: patch-based forecasting with decomposition
- GLCN: "From global to local" (Computers & Electrical Engineering, 2025)
- RoRA: Rotational Rank Adaptation (Sudjianto, 2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import RoRA (optional)
try:
    from layers.rora import PatchRoRA, RoRA
    RORA_AVAILABLE = True
except ImportError:
    RORA_AVAILABLE = False
    print("[xPatch-GL] Warning: RoRA not available. Install layers/rora.py for full functionality.")


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    
    Normalizes input, stores statistics, denormalizes output.
    Handles distribution shift between train and test.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
            
    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        Args:
            x: [B, L, C] or [B, C, L]
            mode: 'norm' to normalize, 'denorm' to denormalize
        """
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
        else:
            raise ValueError(f"Unknown mode: {mode}")


class IntraPatchModule(nn.Module):
    """
    Intra-patch (Local) processing module.
    
    Operates within each patch to capture local temporal patterns.
    Uses xPatch-style fc → conv structure but focused on within-patch processing.
    
    Architecture:
        [B*C, N, P] → fc1 (expand) → GELU → conv (compress) → fc2 → [B*C, N, P]
    """
    
    def __init__(
        self,
        patch_len: int,
        num_patches: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.num_patches = num_patches
        self.hidden_dim = patch_len * expansion_factor
        
        # Expand within each patch
        self.fc1 = nn.Linear(patch_len, self.hidden_dim)
        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(num_patches)
        
        # Depthwise conv: process each patch independently
        # kernel operates across the hidden dimension
        self.conv = nn.Conv1d(
            in_channels=num_patches,
            out_channels=num_patches,
            kernel_size=3,
            padding=1,
            groups=num_patches,  # depthwise = per-patch
        )
        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(num_patches)
        
        # Compress back to patch_len
        self.fc2 = nn.Linear(self.hidden_dim, patch_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*C, N, P] where N=num_patches, P=patch_len
        Returns:
            [B*C, N, P]
        """
        residual = x
        
        # Expand
        x = self.fc1(x)           # [BC, N, hidden]
        x = self.act1(x)
        x = self.bn1(x)
        
        # Depthwise conv within hidden dimension
        x = self.conv(x)          # [BC, N, hidden]
        x = self.act2(x)
        x = self.bn2(x)
        
        # Compress
        x = self.fc2(x)           # [BC, N, P]
        x = self.dropout(x)
        
        return x + residual


class InterPatchModule(nn.Module):
    """
    Inter-patch (Global) processing module.
    
    Captures global temporal dynamics across patches using:
    1. GlobalAvgPool to compress each patch to a scalar
    2. MLP to model inter-patch dependencies
    3. Broadcast back to patch dimension
    
    This is the key insight from GLCN: explicitly model cross-patch relationships
    separately from within-patch processing.
    
    Architecture:
        [B*C, N, P] → GlobalAvgPool → [B*C, N, 1] → MLP → [B*C, N, 1] → broadcast → [B*C, N, P]
    """
    
    def __init__(
        self,
        patch_len: int,
        num_patches: int,
        hidden_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.num_patches = num_patches
        
        # MLP for inter-patch dynamics
        # Input: N (one value per patch after pooling)
        # Output: N (modulation weights for each patch)
        hidden_dim = num_patches * hidden_factor
        
        self.mlp = nn.Sequential(
            nn.Linear(num_patches, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_patches),
            nn.Dropout(dropout),
        )
        
        # Learnable scale for the global modulation
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*C, N, P] where N=num_patches, P=patch_len
        Returns:
            [B*C, N, P] - global modulation applied to input
        """
        # Global average pool across patch dimension
        pooled = x.mean(dim=-1)           # [BC, N]
        
        # MLP to capture inter-patch dependencies
        weights = self.mlp(pooled)        # [BC, N]
        
        # Apply as multiplicative modulation (like SE-Net style)
        weights = torch.sigmoid(weights)  # [BC, N]
        weights = weights.unsqueeze(-1)   # [BC, N, 1]
        
        # Modulate input
        return x * (1 + self.scale * weights)


class GLBlock(nn.Module):
    """
    Global-Local Block: The core component of xPatch-GL.
    
    Combines:
    1. Intra-patch module (local patterns within each patch)
    2. Inter-patch module (global dynamics across patches)
    3. Fusion mechanism (gated or additive)
    
    This explicit separation is the key insight from GLCN.
    """
    
    def __init__(
        self,
        patch_len: int,
        num_patches: int,
        expansion_factor: int = 4,
        inter_hidden_factor: int = 2,
        fusion_type: str = 'gated',  # 'gated', 'add', 'concat'
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.patch_len = patch_len
        
        # Local processing (within patches)
        self.intra = IntraPatchModule(
            patch_len=patch_len,
            num_patches=num_patches,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )
        
        # Global processing (across patches)
        self.inter = InterPatchModule(
            patch_len=patch_len,
            num_patches=num_patches,
            hidden_factor=inter_hidden_factor,
            dropout=dropout,
        )
        
        # Fusion mechanism
        if fusion_type == 'gated':
            # Learnable gate to balance local vs global
            self.gate = nn.Sequential(
                nn.Linear(patch_len * 2, patch_len),
                nn.Sigmoid(),
            )
        elif fusion_type == 'concat':
            # Project concatenated back to original dim
            self.proj = nn.Linear(patch_len * 2, patch_len)
        # 'add' needs no extra parameters
        
        self.norm = nn.LayerNorm(patch_len)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*C, N, P]
        Returns:
            [B*C, N, P]
        """
        # Process in parallel
        local_out = self.intra(x)   # [BC, N, P]
        global_out = self.inter(x)  # [BC, N, P]
        
        # Fuse
        if self.fusion_type == 'gated':
            concat = torch.cat([local_out, global_out], dim=-1)  # [BC, N, 2P]
            gate = self.gate(concat)  # [BC, N, P]
            fused = gate * local_out + (1 - gate) * global_out
        elif self.fusion_type == 'add':
            fused = local_out + global_out
        elif self.fusion_type == 'concat':
            concat = torch.cat([local_out, global_out], dim=-1)
            fused = self.proj(concat)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        # Residual + norm
        return self.norm(x + fused)


class EMADecomposition(nn.Module):
    """
    EMA-based trend-seasonal decomposition.
    
    Uses exponential moving average to extract trend, leaving residual as seasonal.
    This is the proven winner from xPatch experiments.
    """
    
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [B, L, C]
        Returns:
            trend: [B, L, C]
            seasonal: [B, L, C]
        """
        B, L, C = x.shape
        
        # Compute EMA
        trend = torch.zeros_like(x)
        trend[:, 0, :] = x[:, 0, :]
        
        for t in range(1, L):
            trend[:, t, :] = self.alpha * x[:, t, :] + (1 - self.alpha) * trend[:, t-1, :]
        
        seasonal = x - trend
        return trend, seasonal


class TrendHead(nn.Module):
    """
    Simple MLP head for trend prediction.
    
    Downsamples then predicts.
    """
    
    def __init__(self, seq_len: int, pred_len: int, hidden_dim: int = 256):
        super().__init__()
        
        # Downsample factor
        self.ds_factor = max(1, seq_len // 64)
        # Correct calculation: length after t[:, ::ds_factor] slicing
        ds_len = (seq_len - 1) // self.ds_factor + 1
        
        self.net = nn.Sequential(
            nn.Linear(ds_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pred_len),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*C, L]
        Returns:
            [B*C, P]
        """
        # Downsample
        if self.ds_factor > 1:
            x = x[:, ::self.ds_factor]
        return self.net(x)


class NetworkGL(nn.Module):
    """
    xPatch-GL: Global-Local xPatch Network
    
    A fundamentally new architecture combining:
    - xPatch's EMA decomposition and channel independence
    - GLCN's explicit global-local separation
    - Optional RoRA for geometric alignment
    
    Args:
        seq_len: Input sequence length
        pred_len: Prediction horizon
        enc_in: Number of input channels
        patch_len: Length of each patch (default: 16)
        stride: Patch stride (default: 8)
        d_model: Model dimension (default: patch_len)
        num_gl_blocks: Number of GLBlocks to stack (default: 2)
        expansion_factor: Expansion in intra-patch MLP (default: 4)
        fusion_type: How to fuse local and global ('gated', 'add', 'concat')
        dropout: Dropout rate
        use_rora: Whether to use RoRA for geometric alignment
        rora_rank: Rank for RoRA
        ema_alpha: Alpha for EMA decomposition
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        # Patching
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: str = 'end',
        # Model dimensions
        d_model: int = None,
        num_gl_blocks: int = 2,
        expansion_factor: int = 4,
        inter_hidden_factor: int = 2,
        fusion_type: str = 'gated',
        dropout: float = 0.1,
        # RoRA
        use_rora: bool = False,
        rora_rank: int = 4,
        rora_mode: str = 'feature',
        # Decomposition
        ema_alpha: float = 0.2,
        # RevIN
        use_revin: bool = True,
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.d_model = d_model if d_model is not None else patch_len
        self.use_rora = use_rora and RORA_AVAILABLE
        self.use_revin = use_revin
        
        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.padding_layer = nn.ReplicationPad1d((0, stride))
            self.num_patches += 1
        else:
            self.padding_layer = None
            
        # ==================================================================
        # Normalization
        # ==================================================================
        if use_revin:
            self.revin = RevIN(enc_in, affine=True)
        
        # ==================================================================
        # Decomposition
        # ==================================================================
        self.decomp = EMADecomposition(alpha=ema_alpha)
        
        # ==================================================================
        # Seasonal Stream (Global-Local processing)
        # ==================================================================
        
        # Patch embedding (if d_model != patch_len)
        if self.d_model != patch_len:
            self.patch_embed = nn.Linear(patch_len, self.d_model)
        else:
            self.patch_embed = nn.Identity()
        
        # Stack of GLBlocks
        self.gl_blocks = nn.ModuleList([
            GLBlock(
                patch_len=self.d_model,
                num_patches=self.num_patches,
                expansion_factor=expansion_factor,
                inter_hidden_factor=inter_hidden_factor,
                fusion_type=fusion_type,
                dropout=dropout,
            )
            for _ in range(num_gl_blocks)
        ])
        
        # Optional RoRA for geometric alignment
        if self.use_rora:
            from layers.rora import PatchRoRA
            self.rora = PatchRoRA(
                d_model=self.d_model,
                num_patches=self.num_patches,
                rank=rora_rank,
                mode=rora_mode,
            )
        
        # Seasonal prediction head
        self.seasonal_head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.num_patches * self.d_model, pred_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len * 2, pred_len),
        )
        
        # ==================================================================
        # Trend Stream
        # ==================================================================
        self.trend_head = TrendHead(seq_len, pred_len)
        
        # ==================================================================
        # Final fusion
        # ==================================================================
        self.final_fusion = nn.Linear(pred_len * 2, pred_len)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, C] input time series
        Returns:
            [B, P, C] predictions
        """
        B, L, C = x.shape
        
        # ==================================================================
        # RevIN normalization
        # ==================================================================
        if self.use_revin:
            x = self.revin(x, mode='norm')
        
        # ==================================================================
        # Decomposition
        # ==================================================================
        trend, seasonal = self.decomp(x)  # both [B, L, C]
        
        # ==================================================================
        # Seasonal Stream (Global-Local)
        # ==================================================================
        # Reshape for channel independence: [B, L, C] -> [B*C, L]
        s = seasonal.permute(0, 2, 1).reshape(B * C, L)  # [BC, L]
        
        # Padding
        if self.padding_layer is not None:
            s = self.padding_layer(s)
        
        # Patching
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [BC, N, P]
        
        # Patch embedding
        s = self.patch_embed(s)  # [BC, N, d_model]
        
        # GLBlocks
        for gl_block in self.gl_blocks:
            s = gl_block(s)
        
        # RoRA (optional)
        if self.use_rora:
            s = self.rora(s)
        
        # Seasonal prediction
        s_pred = self.seasonal_head(s)  # [BC, pred_len]
        
        # ==================================================================
        # Trend Stream
        # ==================================================================
        t = trend.permute(0, 2, 1).reshape(B * C, L)  # [BC, L]
        t_pred = self.trend_head(t)  # [BC, pred_len]
        
        # ==================================================================
        # Fusion
        # ==================================================================
        combined = torch.cat([s_pred, t_pred], dim=-1)  # [BC, 2*pred_len]
        out = self.final_fusion(combined)  # [BC, pred_len]
        
        # Reshape back: [BC, pred_len] -> [B, C, pred_len] -> [B, pred_len, C]
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)
        
        # ==================================================================
        # RevIN denormalization
        # ==================================================================
        if self.use_revin:
            out = self.revin(out, mode='denorm')
        
        return out
    
    def get_num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Variant: NetworkGL_Lite (even lighter version)
# =============================================================================

class NetworkGL_Lite(nn.Module):
    """
    Lighter version of xPatch-GL with:
    - Single GLBlock
    - No RoRA
    - Simpler heads
    
    For comparison and resource-constrained environments.
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int = 16,
        stride: int = 8,
        expansion_factor: int = 2,
        dropout: float = 0.1,
        ema_alpha: float = 0.2,
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.stride = stride
        
        # Patches
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.padding = nn.ReplicationPad1d((0, stride))
        self.num_patches += 1
        
        # RevIN
        self.revin = RevIN(enc_in)
        
        # Decomposition
        self.decomp = EMADecomposition(alpha=ema_alpha)
        
        # Single GLBlock
        self.gl_block = GLBlock(
            patch_len=patch_len,
            num_patches=self.num_patches,
            expansion_factor=expansion_factor,
            fusion_type='add',  # simpler fusion
            dropout=dropout,
        )
        
        # Heads
        self.seasonal_head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.num_patches * patch_len, pred_len),
        )
        
        self.trend_head = nn.Linear(seq_len, pred_len)
        
        # Fusion
        self.fusion = nn.Linear(pred_len * 2, pred_len)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        
        x = self.revin(x, mode='norm')
        trend, seasonal = self.decomp(x)
        
        # Seasonal
        s = seasonal.permute(0, 2, 1).reshape(B * C, L)
        s = self.padding(s)
        s = s.unfold(-1, self.patch_len, self.stride)
        s = self.gl_block(s)
        s_pred = self.seasonal_head(s)
        
        # Trend
        t = trend.permute(0, 2, 1).reshape(B * C, L)
        t_pred = self.trend_head(t)
        
        # Fuse
        out = self.fusion(torch.cat([s_pred, t_pred], dim=-1))
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)
        out = self.revin(out, mode='denorm')
        
        return out


# =============================================================================
# Factory function
# =============================================================================

def create_network_gl(
    seq_len: int,
    pred_len: int,
    enc_in: int,
    variant: str = 'standard',
    **kwargs
) -> nn.Module:
    """
    Factory function to create xPatch-GL variants.
    
    Args:
        seq_len: Input sequence length
        pred_len: Prediction horizon
        enc_in: Number of channels
        variant: 'standard', 'lite', or 'rora'
        **kwargs: Additional arguments for the network
    
    Returns:
        NetworkGL instance
    """
    if variant == 'lite':
        return NetworkGL_Lite(seq_len, pred_len, enc_in, **kwargs)
    elif variant == 'rora':
        return NetworkGL(seq_len, pred_len, enc_in, use_rora=True, **kwargs)
    else:
        return NetworkGL(seq_len, pred_len, enc_in, **kwargs)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Test the network
    B, L, C = 32, 512, 7
    pred_len = 96
    
    model = NetworkGL(
        seq_len=L,
        pred_len=pred_len,
        enc_in=C,
        patch_len=16,
        stride=8,
        num_gl_blocks=2,
        use_rora=False,
    )
    
    x = torch.randn(B, L, C)
    y = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters:   {model.get_num_params():,}")
    
    # Test with RoRA
    if RORA_AVAILABLE:
        model_rora = NetworkGL(
            seq_len=L,
            pred_len=pred_len,
            enc_in=C,
            use_rora=True,
            rora_rank=4,
        )
        y_rora = model_rora(x)
        print(f"\nWith RoRA:")
        print(f"Output shape: {y_rora.shape}")
        print(f"Parameters:   {model_rora.get_num_params():,}")
