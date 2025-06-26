"""
LoRA (Low-Rank Adaptation) implementations for efficient model adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional


class QwenLoRALayer(nn.Module):
    """
    LoRA implementation for Qwen2.5 models as used in the paper
    Follows exact LoRA formulation from Equation (1) in paper
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 rank: int = 8,
                 alpha: float = 16.0,
                 original_weight: Optional[torch.Tensor] = None,
                 original_bias: Optional[torch.Tensor] = None):
        """Create a LoRA adapter for a linear layer.

        Args:
            in_features:  Size of the input feature dimension (d_in).
            out_features: Size of the output feature dimension (d_out).
            rank:         LoRA rank (r).
            alpha:        LoRA scaling hyper-parameter (α).
            original_weight: Optional tensor holding the *frozen* pretrained
                weight matrix W₀. If *None*, a small random matrix is used as
                placeholder. Passing the real weight is strongly recommended.
            original_bias: Optional tensor with the frozen bias b₀ from the
                original linear layer. If provided, the bias will be applied in
                the forward pass.
        """
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # LoRA scaling factor

        # LoRA matrices A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Frozen (pretrained) weights and optional bias
        if original_weight is None:
            original_weight = torch.randn(out_features, in_features) * 0.02
        self.register_buffer("frozen_weight", original_weight.detach().clone())

        self.has_bias = original_bias is not None
        if self.has_bias:
            self.register_buffer("frozen_bias", original_bias.detach().clone())
        else:
            self.frozen_bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: (W0 + BA)x where ΔW = BA from Equation (1)
        """
        # Original output (W₀x + b₀)
        original_output = F.linear(x, self.frozen_weight, self.frozen_bias)
        
        # LoRA adaptation: ΔW = BA
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        
        return original_output + lora_output
    
    def get_lora_parameters(self) -> torch.Tensor:
        """Get tokenized LoRA parameters as mentioned in Section 2.5"""
        with torch.no_grad():
            return torch.cat([self.lora_A.flatten(), self.lora_B.flatten()]).clone()
    
    def set_lora_parameters(self, params: torch.Tensor):
        """Set LoRA parameters from generated vector"""
        A_size = self.rank * self.lora_A.size(1)
        
        # Reshape back to original matrices
        with torch.no_grad():
            self.lora_A.copy_(params[:A_size].view(self.rank, -1))
            self.lora_B.copy_(params[A_size:].view(self.lora_B.size(0), self.rank))


def get_qwen_lora_configs(lora_rank: int = 8) -> List[Dict]:
    """
    LoRA configurations for Qwen2.5 architecture
    Based on standard transformer layer structure
    """
    # Qwen2.5-0.5B dimensions (approximate)
    hidden_size = 896
    intermediate_size = 4864
    
    configs = []
    
    # Attention layers (12 layers for 0.5B model)
    for layer_idx in range(12):
        configs.extend([
            {
                'name': f'layers.{layer_idx}.self_attn.q_proj',
                'in_features': hidden_size,
                'out_features': hidden_size,
                'rank': lora_rank
            },
            {
                'name': f'layers.{layer_idx}.self_attn.k_proj', 
                'in_features': hidden_size,
                'out_features': hidden_size,
                'rank': lora_rank
            },
            {
                'name': f'layers.{layer_idx}.self_attn.v_proj',
                'in_features': hidden_size,
                'out_features': hidden_size,
                'rank': lora_rank
            },
            {
                'name': f'layers.{layer_idx}.self_attn.o_proj',
                'in_features': hidden_size,
                'out_features': hidden_size,
                'rank': lora_rank
            },
            {
                'name': f'layers.{layer_idx}.mlp.gate_proj',
                'in_features': hidden_size,
                'out_features': intermediate_size,
                'rank': lora_rank
            },
            {
                'name': f'layers.{layer_idx}.mlp.up_proj',
                'in_features': hidden_size,
                'out_features': intermediate_size,
                'rank': lora_rank
            },
            {
                'name': f'layers.{layer_idx}.mlp.down_proj',
                'in_features': intermediate_size,
                'out_features': hidden_size,
                'rank': lora_rank
            }
        ])
    
    return configs 