import torch.nn as nn
import copy
from models.transformer_mlp.encoder_layer import EncoderLayer
from models.transformer_mlp.decoder_layer import DecoderLayer
from models.transformer_mlp.sinusoidal_positional_encoding import SinusoidalPositionalEncoding
from models.transformer_mlp.embedding_positional_encoding import EmbeddingPositionalEncoding


class EncoderDecoder(nn.Module):
    
    def __init__(self, dropout, d_model, type_positional_encoding, positional_encoding_x_now, n_previous_values, n_head, d_hidden, n_encoders, n_decoders, n_features=8):
        
        super(EncoderDecoder, self).__init__()
        
        self.encoder_input = nn.Linear(n_features, d_model)
        self.decoder_input = nn.Linear(n_features, d_model)
        
        if type_positional_encoding is None:
            self.positional_encoding = None
        elif type_positional_encoding == 'Sinusoidal':
            self.positional_encoding = SinusoidalPositionalEncoding(dropout=dropout, max_len_seq=n_previous_values, d_model=d_model)
        elif type_positional_encoding == 'Embedding':
            self.positional_encoding = EmbeddingPositionalEncoding(dropout=dropout, max_len_seq=n_previous_values, d_model=d_model)
            
        self.positional_encoding_x_now = positional_encoding_x_now
        
        encoder_layer = EncoderLayer(dropout=dropout, d_model=d_model, n_head=n_head, d_hidden=d_hidden)
        decoder_layer = DecoderLayer(dropout=dropout, d_model=d_model, n_head=n_head, d_hidden=d_hidden)
        
        self.encoder_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(n_encoders)])
        self.decoder_layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(n_decoders)])
        
    def forward(self, x_now, x_previous):
        
        x_previous = self.encoder_input(x_previous)
        x_now = self.decoder_input(x_now)
        
        if self.positional_encoding is not None:
            x_previous = self.positional_encoding(x_previous)
            if self.positional_encoding_x_now == True:
                x_now = self.positional_encoding(x_now)
            
        for encoder_layer in self.encoder_layers:
            x_previous = encoder_layer(x_previous)
            
        for decoder_layer in self.decoder_layers:
            x_now = decoder_layer(x_now, x_previous)
            
        return x_now