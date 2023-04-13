from hyperparameters_suggestion import get_hyperparameters_suggestion_mlp, get_hyperparameters_suggestion_transformer_mlp


def get_hyperparameters_suggestion(model_name, trial):
    
    if model_name == "MLP":
        get_hyperparameters_suggestion_mlp(trial)
    elif model_name == "TransformerMLP":
        get_hyperparameters_suggestion_transformer_mlp(trial)
    else:
        raise NotImplementedError("Get hyperparameters suggestion not implemented for {}".format(model_name))
        
    