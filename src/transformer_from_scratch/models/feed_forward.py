import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self,d_model, d_ff):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        
        return self.model(x)
    

if __name__=="__main__":
    x = torch.randn(5,8,64)
    
    ffn = FeedForward(64, 256)
    output = ffn(x)
    
    print(output.shape)