import torch
import torch.nn as nn

from transformer_from_scratch.models.multi_head_attention import MultiHeadAttention
from transformer_from_scratch.models.layer_norm import LayerNorm
from transformer_from_scratch.models.feed_forward import FeedForward


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        
        self.feedforward = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        ffn_out = self.feedforward(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x
    
if __name__=="__main__":
    torch.manual_seed(42)

    B = 5
    S = 8
    d_model = 64

    x = torch.randn(B, S, d_model)

    encoder = EncoderBlock(
        d_model=64,
        num_heads=8,
        d_ff=256
    )

    output = encoder(x)

    print(output.shape)

    assert output.shape == (B, S, d_model)
    print("EncoderBlock works!")