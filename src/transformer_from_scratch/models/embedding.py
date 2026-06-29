import math
import torch
import torch.nn as nn
from transformer_from_scratch.datasets.dataloader import get_dataloader


class TokenEmbedding(nn.Module):
    
    def __init__(self, vocab_size, d_model, device='cpu'):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            device=device
        )
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)