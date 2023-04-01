import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate_for_testing(model, dataset_train, dataset_validation, dataset_test, scaler_target, target, optimizer, batch_size_train, batch_size_validation, batch_size_test, learning_rate, weight_decay, patience, epochs):
    
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    model.cuda()
    dataset_train.cuda()
    dataset_validation.cuda()
    dataset_test.cuda()
    
    criterion_mse = nn.MSELoss(reduction='sum')
    criterion_mae = nn.L1Loss(reduction='sum')
    
    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError("Optimizer not correct")
    
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)
    data_loader_validation = DataLoader(dataset_validation, batch_size=batch_size_validation, shuffle=False)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False)
    
    best_validation_loss = np.inf
    counter_no_amelioration = 0
    best_model_state_dict = deepcopy(model.state_dict())
    
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
            best_model_state_dict = deepcopy(model.state_dict())
        else:
            counter_no_amelioration += 1
            if counter_no_amelioration == patience:
                model.load_state_dict(best_model_state_dict)
                break
                
    model.eval()
    loss_test_mse = 0
    loss_test_mae = 0
    loss_test_mse_unscaled = 0
    loss_test_mae_unscaled = 0
    relative_error = 0
    max_error_absolute = 0
    max_error_relative = 0
    
    for x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month, y_bid, y_ask in data_loader_test:
            
        if target == 'y_bid':
            y = y_bid
        elif target == 'y_ask':
            y = y_ask
        else:
            raise NotImplementedError("Target not correct")
                
        with torch.no_grad():
            
            y_pred = model.forward(x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month)
            
            loss_test_mse += criterion_mse(y_pred, y).item()
            loss_test_mae += criterion_mae(y_pred, y).item()
                        
            y_pred = scaler_target.inverse_transform(y_pred.detach().cpu().unsqueeze(dim=1).numpy()).squeeze()
            y = scaler_target.inverse_transform(y.detach().cpu().unsqueeze(dim=1).numpy()).squeeze()
                        
            loss_test_mse_unscaled += mean_squared_error(y, y_pred) * batch_size_test
            loss_test_mae_unscaled += mean_absolute_error(y, y_pred) * batch_size_test
            relative_error += np.sum(np.abs((y - y_pred) / y))
            
            if np.max(np.abs(y - y_pred)) > max_error_absolute:
                max_error_absolute = np.max(np.abs(y - y_pred))
                
            if np.max(np.abs((y - y_pred) / y)) > max_error_relative:
                max_error_relative = np.max(np.abs((y - y_pred) / y))
            
                    
    loss_test_mse = loss_test_mse / dataset_test.__len__()
    loss_test_mae = loss_test_mae / dataset_test.__len__()
    loss_test_mse_unscaled = loss_test_mse_unscaled / dataset_test.__len__()
    loss_test_mae_unscaled = loss_test_mae_unscaled / dataset_test.__len__()
    relative_error = relative_error / dataset_test.__len__()
    
    return model, loss_test_mse, loss_test_mae, loss_test_mse_unscaled, loss_test_mae_unscaled, relative_error, max_error_absolute, max_error_relative