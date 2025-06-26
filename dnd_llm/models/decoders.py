"""
Hyper-convolutional decoders for transforming prompts to parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperConvolutionalBlock(nn.Module):
    """
    Exact implementation of hyper-convolutional block from Figure 3 and Section 2.5
    Each block contains three hyper-convolution modules extracting features in different dimensions
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Three hyper-convolution modules as described in paper
        # ConvW: operates on (C, L) dimension  
        # ConvH: operates on (L, N) dimension
        # ConvL: operates on (N, L) dimension
        
        self.conv1_W = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2_W = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.conv1_H = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2_H = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.conv_L = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Bias term b³ from Equation (5)
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass following Equation (5) in paper:
        c^l_W = Conv2H(Conv1W(c^{l-1}))
        c^l_H = Conv2W(Conv2H(c^{l-1}))  
        c^l = ConvL(c^l_W + c^l_H + b³)
        """
        # Width convolution path
        c_W = F.relu(self.conv1_W(x))
        c_W = self.conv2_W(c_W)
        
        # Height convolution path
        c_H = F.relu(self.conv1_H(x))
        c_H = self.conv2_H(c_H)
        
        # Combine with bias and apply layer-wise convolution
        combined = c_W + c_H + self.bias
        output = self.conv_L(combined)
        
        return F.relu(output)


class CascadedHyperConvolutionalDecoder(nn.Module):
    """
    Cascaded hyper-convolutional decoder from Section 2.5
    Transforms prompt embeddings to LoRA weight dimensions
    """
    def __init__(self, input_dim: int, target_param_count: int, prompt_batch_length: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.target_param_count = target_param_count
        self.prompt_batch_length = prompt_batch_length
        
        # Architecture progression as mentioned in paper appendix
        # Transform [B, N, L, C] -> [B, Nw, Lw, Cw]
        
        # Initial projection to 4D tensor
        self.input_projection = nn.Linear(input_dim, 128 * 32)
        
        # Cascaded blocks with specific channel progressions from Table 8
        channel_configs = [
            (128, 256), (256, 512), (512, 1024), 
            (1024, 2048), (2048, 4096), (4096, 8192)
        ]
        
        self.conv_blocks = nn.ModuleList([
            HyperConvolutionalBlock(in_ch, out_ch) 
            for in_ch, out_ch in channel_configs
        ])
        
        # Adaptive pooling and final projection to match parameter count
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.final_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192 * 64, 4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, target_param_count)
        )
        
    def forward(self, prompt_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Transform prompt embeddings to parameter space
        Input: [B, N, L, C] where N=prompt_batch_length, L=sequence_length, C=embedding_dim
        Output: [B, target_param_count]
        """
        B, N, L, C = prompt_embeddings.shape
        
        # Reshape and project to initial 4D tensor
        x = prompt_embeddings.view(B, N * L, C)
        x = self.input_projection(x)
        x = x.view(B, 128, N, -1)
        
        # Ensure spatial dimensions are appropriate
        if x.size(-1) < 32:
            x = F.interpolate(x, size=(N, 32), mode='bilinear', align_corners=False)
        
        # Pass through cascaded hyper-convolutional blocks
        for block in self.conv_blocks:
            x = block(x)
            # Optional spatial reduction to manage memory
            if x.size(-1) > 16:
                x = F.avg_pool2d(x, kernel_size=2, stride=1, padding=0)
        
        # Final projection to parameter space
        x = self.adaptive_pool(x)
        parameters = self.final_projection(x)
        
        return parameters 