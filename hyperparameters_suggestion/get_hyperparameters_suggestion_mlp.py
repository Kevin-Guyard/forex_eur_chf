

def get_hyperparameters_suggestion_mlp(trial, type_memory, n_hidden_layers):
    
    trial.suggest_int("batch_size_train", low=4, high=9)
    trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    trial.suggest_float("learning_rate", 1e-5, 1e0, log=True)
    trial.suggest_float("weight_decay", 1e-5, 1e0, log=True)
    
    trial.suggest_float("dropout", low=0, high=0.9)
    trial.suggest_int("embedding_dim_year", low=2, high=6)
    trial.suggest_int("embedding_dim_month", low=2, high=7)
    trial.suggest_int("embedding_dim_day", low=4, high=18)
    trial.suggest_int("embedding_dim_hour", low=4, high=15)
    trial.suggest_int("embedding_dim_weekday", low=2, high=4)
    
    if type_memory == 'hour':
        trial.suggest_int("n_previous_hour_values", low=1, high=720)
    elif type_memory == 'day':
        trial.suggest_int("n_previous_hour_values", low=0, high=720)
        trial.suggest_int("n_previous_day_values", low=1, high=90)
    elif type_memory == 'week':
        trial.suggest_int("n_previous_hour_values", low=0, high=720)
        trial.suggest_int("n_previous_day_values", low=0, high=90)
        trial.suggest_int("n_previous_week_values", low=1, high=50)
    elif type_memory == 'month':
        trial.suggest_int("n_previous_hour_values", low=0, high=720)
        trial.suggest_int("n_previous_day_values", low=0, high=90)
        trial.suggest_int("n_previous_week_values", low=0, high=50)
        trial.suggest_int("n_previous_month_values", low=1, high=36)
        
    for i in range(n_hidden_layers):
        trial.suggest_int("d_hidden_layer_" + str(i), low=2, high=13)
    
    if n_hidden_layers > 0:
        trial.suggest_categorical("hidden_layers_structure", ["Activation-Dropout", "Dropout-Activation", "Batchnorm-Activation", "Activation-Batchnorm"])
    