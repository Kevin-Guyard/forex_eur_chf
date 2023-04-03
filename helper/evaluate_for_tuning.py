import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy


def evaluate_for_tuning(model, dataset_train, dataset_validation, target, optimizer, batch_size_train, batch_size_validation, learning_rate, weight_decay, patience, epochs):
    
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    model.cuda()
    dataset_train.cuda()
    dataset_validation.cuda()

    criterion_mse = nn.MSELoss(reduction='sum')

    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError("Optimizer not correct")
    
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)
    data_loader_validation = DataLoader(dataset_validation, batch_size=batch_size_validation, shuffle=False)
    
    best_validation_loss = np.inf
    counter_no_amelioration = 0
    
    for epoch in range(epochs):
        
        model = model.train()
        
        for x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month, y_bid, y_ask in data_loader_train:
            
            if target == 'y_bid':
                y = y_bid
            elif target == 'y_ask':
                y = y_ask
            else:
                raise NotImplementedError("Target not correct")
                
            optimizer.zero_grad(set_to_none=True)
                            
            y_pred = model.forward(x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month)
            criterion_mse(y_pred, y).backward()
            optimizer.step()
            
        model.eval()
        validation_loss = 0
        
        for x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month, y_bid, y_ask in data_loader_validation:
            
            if target == 'y_bid':
                y = y_bid
            elif target == 'y_ask':
                y = y_ask
            else:
                raise NotImplementedError("Target not correct")
                
            with torch.no_grad():
                y_pred = model.forward(x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month)
                validation_loss += criterion_mse(y_pred, y).item()
                    
        validation_loss = validation_loss / dataset_validation.__len__()
                
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            counter_no_amelioration = 0
        else:
            counter_no_amelioration += 1
            if counter_no_amelioration == patience:
                break
                
    model.cpu()
    dataset_train.cpu()
    dataset_validation.cpu()
                
    return best_validation_loss