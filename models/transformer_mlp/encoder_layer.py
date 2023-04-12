import torch.nn as nn


class EncoderLayer(nn.Module):
    
    def __init__(self, dropout, d_model, n_head, d_hidden):
        
        super(EncoderLayer, self).__init__()
        
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )
        
        self.linear_1 = nn.Linear(d_model, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        
    def forward(self, enc_in):
        
        x, _ = self.multihead_attention(enc_in, enc_in, enc_in)
        
        residual = x
        
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        
        x += residual
        enc_out = self.layernorm(x)
        
        return enc_out