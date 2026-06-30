from typing import Any

import torch
import torch.nn as nn
from transformer_from_scratch.models.attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.attention = ScaledDotProductAttention()
        
    def forward(self, x, mask=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        B, S, _ = Q.shape
        Q = Q.view((B, S,  self.num_heads, self.d_k)).transpose(1,2)
        K = K.view((B, S,  self.num_heads, self.d_k)).transpose(1,2)
        V = V.view((B, S,  self.num_heads, self.d_k)).transpose(1,2)
        output, weights = self.attention(Q, K, V, mask)
        output = output.transpose(1,2)
        output = output.contiguous().view((B, S, self.d_model))
        output = self.W_o(output)
        
        return output, weights
        
if __name__ == "__main__":

    torch.manual_seed(42)
    
    B = 2
    S = 5
    d_model = 64
    num_heads = 8

    x = torch.randn(B, S, d_model)

    mha = MultiHeadAttention(d_model, num_heads)

    output, weights = mha(x)

    print(f"Input:              {x.shape}")
    print(f"Output:             {output.shape}")
    print(f"Attention Weights:  {weights.shape}")

    assert output.shape == (B, S, d_model)
    assert weights.shape == (B, num_heads, S, S)

    print("✅ MultiHeadAttention works correctly!")
        