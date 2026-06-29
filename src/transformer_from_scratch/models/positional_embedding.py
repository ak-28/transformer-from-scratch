import math
from typing import Any
import torch.nn as nn
import torch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length = 5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_length, d_model)
        for pos in range(max_seq_length):
            for i in range(int(d_model/2)):
                denominator = math.pow(10000, 2*i/self.d_model)
                pe[pos, 2*i] = math.sin(pos / denominator)
                pe[pos, 2*i+1] = math.cos(pos / denominator)
        self.register_buffer("pe", pe)
        
    def forward(self, embedding):
        B, S, d = embedding.shape
        return self.pe[:S] # type: ignore
                

pos_enc = PositionalEncoding(512)
x = torch.randint(0,16000, (2,5,512))
emb = pos_enc(x)
print(emb.shape)
print(emb)