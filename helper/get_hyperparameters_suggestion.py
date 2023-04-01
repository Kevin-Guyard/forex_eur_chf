

def get_hyperparameters_suggestion(model_name, trial):

    trial.suggest_int("batch_size_train", low=4, high=9)
    trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    trial.suggest_float("learning_rate", 1e-5, 1e0, log=True)
    trial.suggest_float("weight_decay", 1e-5, 1e0, log=True)
    
    if model_name[0:3] == "MLP":
        
        trial.suggest_float("dropout", low=0, high=0.9)
        trial.suggest_int("embedding_dim_year", low=2, high=6)
        trial.suggest_int("embedding_dim_month", low=2, high=7)
        trial.suggest_int("embedding_dim_day", low=4, high=18)
        trial.suggest_int("embedding_dim_hour", low=4, high=15)
        
        if model_name[4:] == "HourMemory":
        
            trial.suggest_int("n_previous_hour_values", low=1, high=720)
            
        elif model_name[4:] == "DayMemory":
    
            trial.suggest_int("n_previous_hour_values", low=0, high=720)
            trial.suggest_int("n_previous_day_values", low=1, high=90)
            
        elif model_name[4:] == "WeekMemory":
    
            trial.suggest_int("n_previous_hour_values", low=0, high=720)
            trial.suggest_int("n_previous_day_values", low=0, high=90)
            trial.suggest_int("n_previous_week_values", low=1, high=50)
            
        elif model_name[4:] == "MonthMemory":
    
            trial.suggest_int("n_previous_hour_values", low=0, high=720)
            trial.suggest_int("n_previous_day_values", low=0, high=90)
            trial.suggest_int("n_previous_week_values", low=0, high=50)
            trial.suggest_int("n_previous_month_values", low=1, high=36)
            
        else:
            
            raise NotImplementedError("Get hyperparameters suggestion not implemented for {}".format(model_name))
            
        for i in range(int(model_name[3])):
            trial.suggest_int("d_hidden_layer_" + str(i), low=5, high=1000, log=True)

    else:
        
        raise NotImplementedError("Get hyperparameters suggestion not implemented for {}".format(model_name))
        
    