from models import MLP, TransformerMLP
import torch
import numpy as np
import random


def get_model(model_name, **params):
    
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
        
    if model_name[0:3] == "MLP":
    
        return MLP(
            dropout=params["dropout"],
            embedding_dims={
                'year': params['embedding_dim_year'],
                'month': params['embedding_dim_month'],
                'day': params['embedding_dim_day'],
                'hour': params['embedding_dim_hour'],
                'weekday': params['embedding_dim_weekday']
            },
            n_previous_values={
                'hour': params.get('n_previous_hour_values', 0),
                'day': params.get('n_previous_day_values', 0),
                'week': params.get('n_previous_week_values', 0),
                'month': params.get('n_previous_month_values', 0)
            },
            d_hidden_layers=[int(2 ** params['d_hidden_layer_' + str(i)]) for i in range(params['n_hidden_layers'])],
            hidden_layers_structure=params.get("hidden_layers_structure", None)
        )
    
    elif model_name[0:14] == "TransformerMLP":
        
        return TransformerMLP(
            dropout=params["dropout"],
            embedding_dims={
                'year': params['embedding_dim_year'],
                'month': params['embedding_dim_month'],
                'day': params['embedding_dim_day'],
                'hour': params['embedding_dim_hour'],
                'weekday': params['embedding_dim_weekday']
            },
            n_previous_values={
                'hour': params.get('n_previous_hour_values', 0),
                'day': params.get('n_previous_day_values', 0),
                'week': params.get('n_previous_week_values', 0),
                'month': params.get('n_previous_month_values', 0)
            },
            type_positional_encoding=params['type_positional_encoding'],
            positional_encoding_x_now=['positional_encoding_x_now'],
            d_model=int(2 ** params['d_model']),
            n_head=int(2 ** params['n_head']),
            d_hidden=int(2 ** params['d_hidden']),
            n_encoders=params['n_encoders'],
            n_decoders=params['n_decoders'],
            d_hidden_layers=[int(2 ** params['d_hidden_layer_' + str(i)]) for i in range(params['n_hidden_layers'])],
            hidden_layers_structure=params.get("hidden_layers_structure", None)
        )
    
    else:
        
        raise NotImplementedError("Get model not implemented for {}".format(model_name))