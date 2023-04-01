from warnings import simplefilter
import optuna
import torch
import pickle
import os
import sys
from helper import objective, ForexDataset


def main():
    
    TUNING_PATIENCE = 5
    TUNING_EPOCHS = 1
    
    args = sys.argv[1:]
    
    simplefilter("ignore")
    #optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    model_name = args[0]
    target = args[1]
    
    dataset_tuning_trains = []
    dataset_tuning_validations = []
    
    for i in range(4):
        
        with open('./data preprocessed/dataset_tuning_train_' + str(i) + '.pt', 'rb') as file:
            dataset_tuning_trains.append(torch.load(file, pickle_module=pickle))
            
        with open('./data preprocessed/dataset_tuning_validation_' + str(i) + '.pt', 'rb') as file:
            dataset_tuning_validations.append(torch.load(file, pickle_module=pickle))
            
    if os.path.exists(os.path.join('studies', target, 'study ' + target + ' ' + model_name + '.pkl')):
        with open(os.path.join('studies', target, 'study ' + target + ' ' + model_name + '.pkl'), 'rb') as file:
            study = pickle.load(file)
    else:
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        
    while True:
    
        study.optimize(
            lambda trial: objective(
                trial, 
                dataset_tuning_trains, 
                dataset_tuning_validations, 
                model_name, 
                target, 
                patience=TUNING_PATIENCE, 
                epochs=TUNING_EPOCHS),
            n_trials=1, 
            timeout=None, 
            n_jobs=1)
        
        with open(os.path.join('studies', target, 'study ' + target + ' ' + model_name + '.pkl'), 'wb') as file:
            pickle.dump(study, file)
    
    
if __name__ == "__main__":
    
    main()