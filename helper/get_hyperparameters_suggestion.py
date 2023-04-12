from hyperparameters_suggestion import get_hyperparameters_suggestion_mlp, get_hyperparameters_suggestion_transformer_mlp


def get_hyperparameters_suggestion(model_name, trial):
    
    if 'HourMemory' in model_name:
        type_memory = 'hour'
    elif 'DayMemory' in model_name:
        type_memory = 'day'
    elif 'WeekMemory' in model_name:
        type_memory = 'week'
    elif 'MonthMemory' in model_name:
        type_memory = 'month'
    
    if model_name[0:3] == "MLP":
        get_hyperparameters_suggestion_mlp(trial, type_memory, int(model_name[3]))
    elif model_name[0:14] == "TransformerMLP":
        get_hyperparameters_suggestion_transformer_mlp(trial, type_memory, int(model_name[14]))
    else:
        raise NotImplementedError("Get hyperparameters suggestion not implemented for {}".format(model_name))
        
    