import math
import torch.nn as nn
import torch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model,dropout=0.1, max_seq_length = 5000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_seq_length, d_model, dtype=torch.float)
        position = torch.arange(max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2, dtype=torch.float) * (-math.log(10000)/d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, embedding):
        _, S, _ = embedding.shape
        return self.dropout(embedding + self.pe[:, :S]) # type: ignore
                

# pos_enc = PositionalEncoding(512)
# x = torch.randint(0,16000, (2,5,512))
# emb = pos_enc(x)
# print(emb.shape)
# print(emb)