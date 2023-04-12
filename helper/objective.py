from helper.get_hyperparameters_suggestion import get_hyperparameters_suggestion
from helper.get_model import get_model 
from helper.cross_evaluate_for_tuning import cross_evaluate_for_tuning

def objective(trial, dataset_tuning_trains, dataset_tuning_validations, model_name, patience, epochs, memory_care=False):
    
    get_hyperparameters_suggestion(model_name, trial)
    
    model = get_model(model_name, **trial.params)
                
    score = cross_evaluate_for_tuning(
        model, 
        dataset_tuning_trains, 
        dataset_tuning_validations, 
        optimizer=trial.params['optimizer'], 
        batch_size_train=int(2 ** trial.params['batch_size_train']), 
        learning_rate=trial.params['learning_rate'], 
        weight_decay=trial.params['weight_decay'], 
        patience=patience, 
        epochs=epochs,
        memory_care=memory_care)
    
    del model
    
    return score