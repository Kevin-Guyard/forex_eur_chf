import torch.nn as nn
import torch
from models.transformer_mlp.encoder_decoder import EncoderDecoder
from models.transformer_mlp.embedding import Embedding


class TransformerMLP(nn.Module):
    
    def __init__(self, dropout, embedding_dims, n_previous_values, type_positional_encoding, positional_encoding_x_now, d_model, n_head, d_hidden, n_encoders, n_decoders, d_hidden_layers, hidden_layers_structure, n_features=8):
        
        super(TransformerMLP, self).__init__()
        
        self.embeddings = Embedding(
            dropout=dropout,
            embedding_dim_year=embedding_dims.get('year'),
            embedding_dim_month=embedding_dims.get('month'),
            embedding_dim_day=embedding_dims.get('day'),
            embedding_dim_hour=embedding_dims.get('hour'),
            embedding_dim_weekday=embedding_dims.get('weekday')
        )
        
        self.n_previous_hour_values = n_previous_values['hour']
        self.n_previous_day_values = n_previous_values['day']
        self.n_previous_week_values = n_previous_values['week']
        self.n_previous_month_values = n_previous_values['month']
        
        n_previous_state = 0
        
        if self.n_previous_hour_values > 0:
            self.encoder_decoder_hour = EncoderDecoder(dropout, d_model, type_positional_encoding, positional_encoding_x_now, self.n_previous_hour_values, n_head, d_hidden, n_encoders, n_decoders)
            n_previous_state += 1
            
        if self.n_previous_day_values > 0:
            self.encoder_decoder_day = EncoderDecoder(dropout, d_model, type_positional_encoding, positional_encoding_x_now, self.n_previous_day_values, n_head, d_hidden, n_encoders, n_decoders)
            n_previous_state += 1
            
        if self.n_previous_week_values > 0:
            self.encoder_decoder_week = EncoderDecoder(dropout, d_model, type_positional_encoding, positional_encoding_x_now, self.n_previous_week_values, n_head, d_hidden, n_encoders, n_decoders)
            n_previous_state += 1
            
        if self.n_previous_month_values > 0:
            self.encoder_decoder_month = EncoderDecoder(dropout, d_model, type_positional_encoding, positional_encoding_x_now, self.n_previous_month_values, n_head, d_hidden, n_encoders, n_decoders)
            n_previous_state += 1
            
        layers_size = [embedding_dims['year'] + embedding_dims['month'] + embedding_dims['day'] + embedding_dims['hour'] + embedding_dims['weekday'] + (n_previous_state) * d_model] + d_hidden_layers
        
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
        
        x_date = self.embeddings(x_date)
        x_now = x_now.unsqueeze(dim=1)
        
        x = [x_date]
        
        if self.n_previous_hour_values > 0:
            x_previous_hour = x_previous_hour[:, 0:self.n_previous_hour_values, :]
            x_now_attention_hour = self.encoder_decoder_hour(x_now, x_previous_hour).squeeze(dim=1)
            x.append(x_now_attention_hour)
            
        if self.n_previous_day_values > 0:
            x_previous_day = x_previous_day[:, 0:self.n_previous_day_values, :]
            x_now_attention_day = self.encoder_decoder_day(x_now, x_previous_day).squeeze(dim=1)
            x.append(x_now_attention_day)
            
        if self.n_previous_week_values > 0:
            x_previous_week = x_previous_week[:, 0:self.n_previous_week_values, :]
            x_now_attention_week = self.encoder_decoder_week(x_now, x_previous_week).squeeze(dim=1)
            x.append(x_now_attention_week)
            
        if self.n_previous_month_values > 0:
            x_previous_month = x_previous_month[:, 0:self.n_previous_month_values, :]
            x_now_attention_month = self.encoder_decoder_month(x_now, x_previous_month).squeeze(dim=1)
            x.append(x_now_attention_month)
        
        x = torch.cat(x, dim=1)
        
        y = self.mlp(x)
        
        y_bid = torch.flatten(y[:, 0])
        y_ask = torch.flatten(y[:, 1])
        
        return y_bid, y_ask