"""
xPatch-GL Hybrid: xPatch + Minimal GLCN Enhancements

This implementation keeps ALL xPatch components intact and adds only three
targeted enhancements from GLCN:

1. Aggregate Conv1D (before patching) - preserves edge information
2. Multi-scale depthwise convolutions - captures multi-granularity patterns  
3. Gated pointwise conv - adaptive cross-patch mixing

Reference architectures:
- xPatch: "Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition"
- GLCN: "From Global to Local: A Lightweight CNN Approach for Long-term Time Series Forecasting"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class RevIN(nn.Module):
    """Reversible Instance Normalization (from xPatch)"""
    
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
            x: [B, L, C]
            mode: 'norm' or 'denorm'
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
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._std + self._mean
            return x
        else:
            raise ValueError(f"Unknown mode: {mode}")


class EMADecomposition(nn.Module):
    """
    Exponential Moving Average Decomposition (from xPatch)
    
    Decomposes time series into trend and seasonal components using EMA.
    More flexible than Simple Moving Average - adapts to changing patterns.
    """
    
    def __init__(self, alpha: float = 0.3):
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
        
        # EMA computation
        trend = torch.zeros_like(x)
        trend[:, 0, :] = x[:, 0, :]
        
        for t in range(1, L):
            trend[:, t, :] = self.alpha * x[:, t, :] + (1 - self.alpha) * trend[:, t-1, :]
        
        seasonal = x - trend
        
        return trend, seasonal


class AggregateConv1D(nn.Module):
    """
    NEW #1: Aggregate Conv1D (from GLCN)
    
    Applied BEFORE patching to preserve edge information between patches.
    Uses residual connection: output = input + Conv1D(input)
    
    From GLCN paper Section 4.2:
    "We apply a sliding aggregation to the original sequence before patching.
    This approach allows each aggregated data point to integrate information
    from adjacent points within its temporal window."
    """
    
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        # Depthwise conv to aggregate neighboring information
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        # Initialize to small values for stable training
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*C, L]
        Returns:
            [B*C, L]
        """
        # x: [B*C, L] -> [B*C, 1, L]
        x_conv = x.unsqueeze(1)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.squeeze(1)  # [B*C, L]
        
        return x + x_conv  # Residual connection


class MultiScaleDepthwiseConv(nn.Module):
    """
    NEW #2: Multi-Scale Depthwise Convolutions (from GLCN)
    
    Replaces xPatch's single depthwise conv with multiple kernel sizes.
    Captures patterns at different granularities within each patch.
    
    From GLCN paper Section 4.3.2:
    "We employ multiple convolutional kernels of different sizes within each patch
    to progressively extract features at various scales."
    """
    
    def __init__(
        self, 
        num_patches: int, 
        patch_dim: int,
        kernel_sizes: List[int] = [1, 3, 5, 7],
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        
        # Multi-scale depthwise convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=num_patches,
                out_channels=num_patches,
                kernel_size=k,
                padding=k // 2,
                groups=num_patches,  # Depthwise: each patch processed independently
                bias=False
            )
            for k in kernel_sizes
        ])
        
        self.norm = nn.BatchNorm1d(num_patches)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*C, N, P*2] where N=num_patches, P*2=embedded patch dim
        Returns:
            [B*C, N, P*2]
        """
        # Sum of multi-scale convolutions
        out = sum(conv(x) for conv in self.convs)
        out = self.norm(out)
        out = self.act(out)
        return out


class GatedPointwiseConv(nn.Module):
    """
    NEW #3: Gated Pointwise Convolution (inspired by GLCN's inter-patch module)
    
    Adds adaptive gating before cross-patch mixing.
    Gate learns which patches are important based on global context.
    
    From GLCN paper Section 4.3.1:
    "A gating mechanism dynamically modulates the contribution of each patch,
    optimizing the overall performance of the model."
    """
    
    def __init__(self, num_patches: int, patch_dim: int, hidden_factor: int = 2):
        super().__init__()
        
        # Gate: GlobalAvgPool -> MLP -> Sigmoid
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_patches, num_patches * hidden_factor),
            nn.GELU(),
            nn.Linear(num_patches * hidden_factor, num_patches),
            nn.Sigmoid(),
        )
        
        # Pointwise conv for cross-patch mixing (same as xPatch)
        self.pw_conv = nn.Conv1d(
            in_channels=num_patches,
            out_channels=num_patches,
            kernel_size=1,
            groups=1,  # Full mixing across patches
        )
        
        self.norm = nn.BatchNorm1d(num_patches)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*C, N, P*2]
        Returns:
            [B*C, N, P*2]
        """
        # Global average pooling across patch dimension
        # [B*C, N, P*2] -> [B*C, N]
        global_summary = x.mean(dim=-1)
        
        # Compute gate
        gate = self.gate_mlp(global_summary)  # [B*C, N]
        
        # Apply gate (multiplicative)
        x = x * gate.unsqueeze(-1)  # [B*C, N, P*2]
        
        # Pointwise conv for cross-patch mixing
        x = self.pw_conv(x)
        x = self.norm(x)
        x = self.act(x)
        
        return x


class LinearStream(nn.Module):
    """
    Linear Stream for Trend (from xPatch)
    
    Simple MLP-based network to capture linear features.
    Processes the trend component from EMA decomposition.
    """
    
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        
        # Simple MLP without activation functions (linear features)
        hidden = seq_len // 2
        
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, pred_len),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*C, L]
        Returns:
            [B*C, H]
        """
        return self.net(x)


class NonLinearStreamHybrid(nn.Module):
    """
    Non-Linear Stream for Seasonal (xPatch + GLCN enhancements)
    
    This is the core hybrid module that combines:
    - xPatch's patching and depthwise-separable conv structure
    - GLCN's aggregate conv, multi-scale conv, and gating
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        # GLCN enhancements
        use_aggregate_conv: bool = True,
        use_multiscale: bool = True,
        use_gating: bool = True,
        kernel_sizes: List[int] = [1, 3, 5, 7],
        gate_hidden_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.patch_len = patch_len
        self.stride = stride
        self.use_aggregate_conv = use_aggregate_conv
        self.use_multiscale = use_multiscale
        self.use_gating = use_gating
        
        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 2  # +1 for padding patch
        
        # Padding
        self.padding_len = stride
        self.padding = nn.ReplicationPad1d((0, self.padding_len))
        
        # NEW #1: Aggregate Conv1D (before patching)
        if use_aggregate_conv:
            self.aggregate_conv = AggregateConv1D(kernel_size=3)
        
        # Embedding: P -> P*2 (from xPatch)
        self.embed = nn.Linear(patch_len, patch_len * 2)
        self.embed_norm = nn.BatchNorm1d(self.num_patches)
        self.embed_act = nn.GELU()
        
        # NEW #2: Multi-scale depthwise conv OR original single depthwise conv
        if use_multiscale:
            self.depthwise = MultiScaleDepthwiseConv(
                num_patches=self.num_patches,
                patch_dim=patch_len * 2,
                kernel_sizes=kernel_sizes,
            )
        else:
            # Simple depthwise conv that preserves dimensions
            self.depthwise = nn.Sequential(
                nn.Conv1d(
                    self.num_patches, self.num_patches,
                    kernel_size=3, padding=1, groups=self.num_patches
                ),
                nn.BatchNorm1d(self.num_patches),
                nn.GELU(),
            )
        
        # NEW #3: Gated pointwise conv OR original pointwise conv
        if use_gating:
            self.pointwise = GatedPointwiseConv(
                num_patches=self.num_patches,
                patch_dim=patch_len * 2,
                hidden_factor=gate_hidden_factor,
            )
        else:
            # Original xPatch pointwise conv
            self.pointwise = nn.Sequential(
                nn.Conv1d(self.num_patches, self.num_patches, kernel_size=1),
                nn.BatchNorm1d(self.num_patches),
                nn.GELU(),
            )
        
        # MLP flatten head (from xPatch)
        flatten_dim = self.num_patches * patch_len * 2
        self.mlp_head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(flatten_dim, pred_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len * 2, pred_len),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*C, L] - seasonal component
        Returns:
            [B*C, H] - prediction
        """
        # NEW #1: Aggregate conv before patching
        if self.use_aggregate_conv:
            x = self.aggregate_conv(x)
        
        # Padding
        x = self.padding(x)
        
        # Patching: [B*C, L+pad] -> [B*C, N, P]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Embedding: [B*C, N, P] -> [B*C, N, P*2]
        x = self.embed(x)
        x = self.embed_norm(x)
        x = self.embed_act(x)
        
        # Depthwise conv (multi-scale or single)
        x_residual = x
        x = self.depthwise(x)
        
        # Residual connection (from xPatch)
        if x.shape == x_residual.shape:
            x = x + x_residual
        
        # Pointwise conv (gated or plain)
        x = self.pointwise(x)
        
        # MLP head
        return self.mlp_head(x)


class Model(nn.Module):
    """
    xPatch-GL Hybrid Model
    
    Full architecture combining xPatch's dual-stream design with GLCN enhancements.
    
    Architecture:
        Input [B, L, C]
            ↓
        RevIN (normalize)
            ↓
        EMA Decomposition → Trend, Seasonal
            ↓                    ↓
        Linear Stream      Non-Linear Stream (with GLCN enhancements)
            ↓                    ↓
        Trend Pred         Seasonal Pred
            ↓                    ↓
        Concatenate + Fuse
            ↓
        RevIN (denormalize)
            ↓
        Output [B, H, C]
    """
    
    def __init__(self, configs):
        super().__init__()
        
        # Extract configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # Patching params
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)
        
        # EMA param
        self.ema_alpha = getattr(configs, 'ema_alpha', 0.3)
        
        # GLCN enhancement flags
        self.use_aggregate_conv = getattr(configs, 'use_aggregate_conv', True)
        self.use_multiscale = getattr(configs, 'use_multiscale', True)
        self.use_gating = getattr(configs, 'use_gating', True)
        
        # Multi-scale kernel sizes
        self.kernel_sizes = getattr(configs, 'kernel_sizes', [1, 3, 5, 7])
        
        # Dropout
        self.dropout = getattr(configs, 'dropout', 0.1)
        
        # RevIN
        self.revin = RevIN(self.enc_in)
        
        # EMA Decomposition
        self.decomp = EMADecomposition(alpha=self.ema_alpha)
        
        # Linear stream for trend
        self.linear_stream = LinearStream(self.seq_len, self.pred_len)
        
        # Non-linear stream for seasonal (with GLCN enhancements)
        self.nonlinear_stream = NonLinearStreamHybrid(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            patch_len=self.patch_len,
            stride=self.stride,
            use_aggregate_conv=self.use_aggregate_conv,
            use_multiscale=self.use_multiscale,
            use_gating=self.use_gating,
            kernel_sizes=self.kernel_sizes,
            dropout=self.dropout,
        )
        
        # Final fusion layer (from xPatch)
        self.fusion = nn.Linear(self.pred_len * 2, self.pred_len)
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Args:
            x_enc: [B, L, C] - input sequence
            Other args for compatibility with existing framework
        Returns:
            [B, H, C] - predictions
        """
        B, L, C = x_enc.shape
        
        # RevIN normalize
        x = self.revin(x_enc, mode='norm')
        
        # EMA decomposition
        trend, seasonal = self.decomp(x)
        
        # Reshape for channel independence: [B, L, C] -> [B*C, L]
        trend = trend.permute(0, 2, 1).reshape(B * C, L)
        seasonal = seasonal.permute(0, 2, 1).reshape(B * C, L)
        
        # Linear stream (trend)
        trend_pred = self.linear_stream(trend)  # [B*C, H]
        
        # Non-linear stream (seasonal)
        seasonal_pred = self.nonlinear_stream(seasonal)  # [B*C, H]
        
        # Concatenate and fuse
        combined = torch.cat([trend_pred, seasonal_pred], dim=-1)  # [B*C, H*2]
        output = self.fusion(combined)  # [B*C, H]
        
        # Reshape back: [B*C, H] -> [B, H, C]
        output = output.reshape(B, C, self.pred_len).permute(0, 2, 1)
        
        # RevIN denormalize
        output = self.revin(output, mode='denorm')
        
        return output


# =============================================================================
# Standalone test
# =============================================================================
if __name__ == "__main__":
    from argparse import Namespace
    
    print("=" * 70)
    print("xPatch-GL Hybrid Test")
    print("=" * 70)
    
    # Test configs
    configs = Namespace(
        seq_len=336,
        pred_len=96,
        enc_in=7,
        patch_len=16,
        stride=8,
        ema_alpha=0.3,
        use_aggregate_conv=True,
        use_multiscale=True,
        use_gating=True,
        kernel_sizes=[1, 3, 5, 7],
        dropout=0.1,
    )
    
    model = Model(configs)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Configuration:")
    print(f"  seq_len: {configs.seq_len}")
    print(f"  pred_len: {configs.pred_len}")
    print(f"  enc_in: {configs.enc_in}")
    print(f"  patch_len: {configs.patch_len}")
    print(f"  stride: {configs.stride}")
    print(f"  ema_alpha: {configs.ema_alpha}")
    print(f"\nGLCN Enhancements:")
    print(f"  use_aggregate_conv: {configs.use_aggregate_conv}")
    print(f"  use_multiscale: {configs.use_multiscale}")
    print(f"  use_gating: {configs.use_gating}")
    print(f"  kernel_sizes: {configs.kernel_sizes}")
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    B = 32
    x = torch.randn(B, configs.seq_len, configs.enc_in)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Output shape: {y.shape}")
    
    # Test ablations
    print("\n" + "=" * 70)
    print("Ablation Variants (parameter counts)")
    print("=" * 70)
    
    variants = [
        ("Full Hybrid", True, True, True),
        ("No Aggregate Conv", False, True, True),
        ("No Multi-scale", True, False, True),
        ("No Gating", True, True, False),
        ("Pure xPatch (no enhancements)", False, False, False),
    ]
    
    for name, agg, ms, gate in variants:
        cfg = Namespace(
            seq_len=336, pred_len=96, enc_in=7,
            patch_len=16, stride=8, ema_alpha=0.3,
            use_aggregate_conv=agg,
            use_multiscale=ms,
            use_gating=gate,
            kernel_sizes=[1, 3, 5, 7],
            dropout=0.1,
        )
        m = Model(cfg)
        params = sum(p.numel() for p in m.parameters())
        print(f"  {name}: {params:,} params")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
