# models/xpatch_gl.py
"""
xPatch-GL Model Wrapper

Integrates NetworkGL into the training framework.
"""

import torch
import torch.nn as nn
from layers.network_gl import NetworkGL, NetworkGL_Lite


class Model(nn.Module):
    """
    xPatch-GL Model wrapper for training integration.
    
    This follows the same interface as other models in the codebase,
    allowing seamless integration with existing training scripts.
    """
    
    def __init__(self, configs):
        super().__init__()
        
        # Determine variant
        variant = getattr(configs, 'gl_variant', 'standard')
        
        if variant == 'lite':
            self.model = NetworkGL_Lite(
                seq_len=configs.seq_len,
                pred_len=configs.pred_len,
                enc_in=configs.enc_in,
                patch_len=getattr(configs, 'patch_len', 16),
                stride=getattr(configs, 'stride', 8),
                expansion_factor=getattr(configs, 'expansion_factor', 2),
                dropout=getattr(configs, 'dropout', 0.1),
                ema_alpha=getattr(configs, 'ema_alpha', 0.2),
            )
        else:
            self.model = NetworkGL(
                # Core dimensions
                seq_len=configs.seq_len,
                pred_len=configs.pred_len,
                enc_in=configs.enc_in,
                # Patching
                patch_len=getattr(configs, 'patch_len', 16),
                stride=getattr(configs, 'stride', 8),
                padding_patch=getattr(configs, 'padding_patch', 'end'),
                # Model dimensions
                d_model=getattr(configs, 'd_model', None),
                num_gl_blocks=getattr(configs, 'num_gl_blocks', 2),
                expansion_factor=getattr(configs, 'expansion_factor', 4),
                inter_hidden_factor=getattr(configs, 'inter_hidden_factor', 2),
                fusion_type=getattr(configs, 'fusion_type', 'gated'),
                dropout=getattr(configs, 'dropout', 0.1),
                # RoRA
                use_rora=getattr(configs, 'use_rora', False),
                rora_rank=getattr(configs, 'rora_rank', 4),
                rora_mode=getattr(configs, 'rora_mode', 'feature'),
                # Decomposition
                ema_alpha=getattr(configs, 'ema_alpha', 0.2),
                # RevIN
                use_revin=getattr(configs, 'use_revin', True),
            )
            
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Standard forward interface for time series models.
        
        Args:
            x_enc: [B, L, C] input time series
            x_mark_enc: [B, L, M] time features (ignored in xPatch-GL)
            Others: Not used, for interface compatibility
            
        Returns:
            [B, pred_len, C] predictions
        """
        return self.model(x_enc)
    
    def get_num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Convenience functions
# =============================================================================

def create_model(configs):
    """Factory function to create xPatch-GL model."""
    return Model(configs)
