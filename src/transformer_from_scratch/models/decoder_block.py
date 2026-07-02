import torch
import torch.nn as nn

from transformer_from_scratch.models.multi_head_attention import MultiHeadAttention
from transformer_from_scratch.models.layer_norm import LayerNorm
from transformer_from_scratch.models.feed_forward import FeedForward


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model,num_heads)
        self.norm1 = LayerNorm(d_model)
        
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        
        self.feedforward = FeedForward(d_model, d_ff)
        self.norm3 = LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, encoder_mask, decoder_mask):
        
        d_attn, _ = self.self_attention(query, mask=decoder_mask)
        query = self.norm1(query + self.dropout(d_attn))
        
        c_attn, _ = self.cross_attention(query, key, value, mask=encoder_mask)
        output = self.norm2(query + self.dropout(c_attn))
        
        ffn_out = self.feedforward(output)
        output = self.norm3(output + self.dropout(ffn_out))
        
        return output
    
    
if __name__=="__main__":
    
    B = 5
    S = 8
    d_model = 64
    num_heads = 8
    d_ff = 256
    
    query = torch.randn(B, S, d_model)
    key = torch.randn(B, S, d_model)
    value = torch.randn(B, S, d_model)
    
    
    encoder_mask = torch.ones(B,1,1,S)
    decoder_mask = torch.tril(torch.ones(S, S))
    decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(0)

    
    
    decoder = DecoderBlock(d_model, num_heads, d_ff)
    
    output = decoder(query, key, value, encoder_mask, decoder_mask)
    
    print("output shape: ", output.shape)
    
    assert output.shape == (B, S, d_model)
    print("Decoder Block Works!")

        