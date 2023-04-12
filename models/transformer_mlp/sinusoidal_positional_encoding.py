import torch.nn as nn
import torch
import math


class SinusoidalPositionalEncoding(nn.Module):
    
    def __init__(self, dropout, max_len_seq, d_model):
        
        super(SinusoidalPositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
                
        pe = torch.zeros(max_len_seq, d_model)
        position = torch.arange(0, max_len_seq).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        
        return x