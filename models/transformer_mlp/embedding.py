import torch.nn as nn
import torch


class Embedding(nn.Module):
    
    def __init__(self, dropout, embedding_dim_year, embedding_dim_month, embedding_dim_day, embedding_dim_hour, embedding_dim_weekday):
        
        super(Embedding, self).__init__()
        
        self.embedding_year = nn.Embedding(num_embeddings=11, embedding_dim=embedding_dim_year)
        self.embedding_month = nn.Embedding(num_embeddings=12, embedding_dim=embedding_dim_month)
        self.embedding_day = nn.Embedding(num_embeddings=31, embedding_dim=embedding_dim_day)
        self.embedding_hour = nn.Embedding(num_embeddings=24, embedding_dim=embedding_dim_hour)
        self.embedding_weekday = nn.Embedding(num_embeddings=6, embedding_dim=embedding_dim_weekday)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        
        return self.dropout(
            torch.cat([
                self.embedding_year(x[:, 0]),
                self.embedding_month(x[:, 1]),
                self.embedding_day(x[:, 2]),
                self.embedding_hour(x[:, 3]),
                self.embedding_weekday(x[:, 4])
            ], dim=1))