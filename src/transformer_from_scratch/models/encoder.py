import torch
import torch.nn as nn
from transformer_from_scratch.models.encoder_block import EncoderBlock


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x,mask)
            
        return x
            
            
if __name__=="__main__":
    torch.manual_seed(42)

    B = 5
    S = 8
    d_model = 64

    x = torch.randn(B, S, d_model)
    
    encoder = Encoder(
        d_model=64,
        num_heads=8,
        d_ff=256,
        num_layers=6
    )
    
    output = encoder(x)

    print(output.shape)

    assert output.shape == (B, S, d_model)
    print("Encoder works!")