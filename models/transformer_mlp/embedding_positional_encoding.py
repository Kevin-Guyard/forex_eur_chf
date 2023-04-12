import torch.nn as nn
import torch


class EmbeddingPositionalEncoding(nn.Module):
    
    def __init__(self, dropout, max_len_seq, d_model):
        
        super(EmbeddingPositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(num_embeddings=max_len_seq, embedding_dim=d_model)
        positions = torch.arange(max_len_seq)
        self.register_buffer('positions', positions)
        
    def forward(self, x):
        
        positions_info = self.pe(self.positions)
        
        x = x + positions_info
        x = self.dropout(x)
        
        return x