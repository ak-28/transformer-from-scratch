import torch
import torch.nn as nn
from transformer_from_scratch.models.decoder_block import DecoderBlock


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
    def forward(self, query, key, value , encoder_mask, decoder_mask):
        for layer in self.layers:
            query = layer(query, key, value, encoder_mask, decoder_mask)
            
        return query
            
            
if __name__=="__main__":
    torch.manual_seed(42)

    B = 5
    S = 8
    d_model = 64

    query = torch.randn(B, S, d_model)
    key = torch.randn(B, S, d_model)
    value = torch.randn(B, S, d_model)
    
    encoder_mask = torch.ones(B,1,1,S)
    decoder_mask = torch.tril(torch.ones(S, S))
    decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(0)

    
    decoder = Decoder(
        d_model=64,
        num_heads=8,
        d_ff=256,
        num_layers=6
    )
    
    output = decoder(query, key, value, encoder_mask, decoder_mask)

    print(output.shape)

    assert output.shape == (B, S, d_model)
    print("Encoder works!")