import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x):
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        x_var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - x_mean)/torch.sqrt(x_var + self.eps)
        
        x_out = self.gamma * x_norm + self.beta
        return x_out
        

if __name__=="__main__":
    torch.manual_seed(44)
    
    x = torch.randn(5,8,64)
    
    ln_custom = LayerNorm(64)
    x_norm_custom = ln_custom(x)
    
    ln_official = nn.LayerNorm(64)
    x_norm_official = ln_official(x)
    
    print("is custom same as official: ", torch.allclose(x_norm_custom, x_norm_official, atol=1e-5))