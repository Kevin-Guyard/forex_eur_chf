import torch
import torch.nn as nn


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
    
class MLP(nn.Module):
    
    def __init__(self, dropout, embedding_dims, n_previous_values, d_hidden_layers, hidden_layers_structure, n_features=8):
        
        super(MLP, self).__init__()
        
        self.n_previous_hour_values = n_previous_values['hour']
        self.n_previous_day_values = n_previous_values['day']
        self.n_previous_week_values = n_previous_values['week']
        self.n_previous_month_values = n_previous_values['month']
        
        self.embeddings = Embedding(
            dropout=dropout, 
            embedding_dim_year=embedding_dims['year'], 
            embedding_dim_month=embedding_dims['month'], 
            embedding_dim_day=embedding_dims['day'], 
            embedding_dim_hour=embedding_dims['hour'],
            embedding_dim_weekday=embedding_dims['weekday'])
        
        layers_size = [embedding_dims['year'] + embedding_dims['month'] + embedding_dims['day'] + embedding_dims['hour'] + embedding_dims['weekday'] + (self.n_previous_hour_values + self.n_previous_day_values + self.n_previous_week_values + self.n_previous_month_values + 1) * n_features] + d_hidden_layers
        
        if hidden_layers_structure == "Batchnorm-Activation":
                        
            layers = [
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    nn.ReLU()
                )
                for in_features, out_features in zip(
                    layers_size[:-1],
                    layers_size[1:]
                )
            ]
            
        elif hidden_layers_structure == "Activation-Batchnorm":
                        
            layers = [
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_features)
                )
                for in_features, out_features in zip(
                    layers_size[:-1],
                    layers_size[1:]
                )
            ]
            
        elif hidden_layers_structure == "Activation-Dropout":
                        
            layers = [
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                for in_features, out_features in zip(
                    layers_size[:-1],
                    layers_size[1:]
                )
            ]
            
        elif hidden_layers_structure == "Dropout-Activation":
                        
            layers = [
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.Dropout(dropout),
                    nn.ReLU()
                )
                for in_features, out_features in zip(
                    layers_size[:-1],
                    layers_size[1:]
                )
            ]
            
        else:
            layers = []
            
        layers += [nn.Linear(layers_size[-1], 2)]
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month):
        
        x = torch.cat([
            self.embeddings(x_date),
            x_now,
            torch.flatten(x_previous_hour[:, 0:self.n_previous_hour_values, :], start_dim=1, end_dim=-1),
            torch.flatten(x_previous_day[:, 0:self.n_previous_day_values, :], start_dim=1, end_dim=-1),
            torch.flatten(x_previous_week[:, 0:self.n_previous_week_values, :], start_dim=1, end_dim=-1),
            torch.flatten(x_previous_month[:, 0:self.n_previous_month_values, :], start_dim=1, end_dim=-1)
        ], dim=1)
                        
        y = self.mlp(x)
        
        y_bid = torch.flatten(y[:, 0])
        y_ask = torch.flatten(y[:, 1])
        
        return y_bid, y_ask