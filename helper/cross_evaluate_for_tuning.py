from copy import deepcopy
from helper.evaluate_for_tuning import evaluate_for_tuning
import numpy as np


def cross_evaluate_for_tuning(model, dataset_trains, dataset_validations, optimizer, batch_size_train, learning_rate, weight_decay, patience, epochs):
    
    model_state_dict = deepcopy(model.state_dict())
    cross_val_scores = []
    
    for dataset_train, dataset_validation in zip(dataset_trains, dataset_validations):
        
        model.load_state_dict(model_state_dict)
        
        cross_val_scores.append(
            evaluate_for_tuning(
                model, 
                dataset_train, 
                dataset_validation, 
                optimizer, 
                batch_size_train, 
                batch_size_validation=dataset_validation.__len__(), 
                learning_rate=learning_rate, 
                weight_decay=weight_decay, 
                patience=patience, 
                epochs=epochs))
                
    return np.mean(cross_val_scores)