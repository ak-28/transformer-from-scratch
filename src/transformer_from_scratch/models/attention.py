from typing import Any

import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        
        
        
    def forward(self, Q, K, V, mask=None):
        dk = Q.shape[-1]
        scores = Q @ K.transpose(-2,-1)
        scores /= math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))
        attention = self.softmax(scores)
        output = attention @ V
        return output, attention
    
if __name__ == "__main__":

    torch.manual_seed(42)

    B = 2          # Batch size
    S = 5          # Sequence length
    d_k = 64       # Head dimension

    Q = torch.randn(B, S, d_k)
    K = torch.randn(B, S, d_k)
    V = torch.randn(B, S, d_k)

    attention = ScaledDotProductAttention()

    output, weights = attention(Q, K, V)

    print("=" * 50)
    print("Input Shapes")
    print("=" * 50)
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}")

    print("\n" + "=" * 50)
    print("Output Shapes")
    print("=" * 50)
    print(f"Attention Weights: {weights.shape}")
    print(f"Output: {output.shape}")

    print("\n" + "=" * 50)
    print("Row Sum Check (Should be 1)")
    print("=" * 50)
    print(weights.sum(dim=-1))