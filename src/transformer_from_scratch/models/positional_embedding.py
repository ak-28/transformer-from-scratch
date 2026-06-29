import math
from typing import Any
import torch.nn as nn
import torch


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.d_model = d_model
        
    def forward(self, embedding):
        B, S, d = embedding.shape
        out_tensor = torch.zeros((S,d),device=embedding.device,dtype=torch.float32)
        for pos in range(S):
            for i in range(int(d/2)):
                denominator = math.pow(10000, 2*i/self.d_model)
                out_tensor[pos, 2*i] = math.sin(pos / denominator)
                out_tensor[pos, 2*i+1] = math.cos(pos / denominator)
                
        return out_tensor
                

pos_embd = PositionalEmbedding(512)
x = torch.randint(0,16000, (2,5,512))
emb = pos_embd(x)
print(emb.shape)
print(emb)