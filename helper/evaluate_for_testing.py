import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate_for_testing(model, dataset_train, dataset_validation, dataset_test, scaler_y_bid, scaler_y_ask, target, optimizer, batch_size_train, batch_size_validation, batch_size_test, learning_rate, weight_decay, patience, epochs):
    
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    model.cuda()
    dataset_train.cuda()
    dataset_validation.cuda()
    dataset_test.cuda()
    
    criterion_mse = nn.MSELoss(reduction='sum')
    criterion_mae = nn.L1Loss(reduction='sum')
    criterion = nn.MSELoss(reduction='sum')

    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError("Optimizer not correct")
    
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)
    data_loader_validation = DataLoader(dataset_validation, batch_size=batch_size_validation, shuffle=False)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False)
    
    best_validation_loss_mse = np.inf
    best_epoch = 0
    counter_no_amelioration = 0
    best_model_state_dict = deepcopy(model.state_dict())
    
    training_loss_mse, training_loss_mae, validation_loss_mse, validation_loss_mae = 0, 0, 0, 0
    results = {
        '2. MSE training normalized': np.inf,
        '2. MAE training normalized': np.inf,
        '3. MSE validation normalized': np.inf,
        '3. MAE validation normalized': np.inf
    }
    
    for epoch in range(1, epochs + 1):
        
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
            criterion(y_pred, y).backward()
            optimizer.step()
            
            training_loss_mse += criterion_mse(y_pred, y).item()
            training_loss_mae += criterion_mae(y_pred, y).item()
            
        training_loss_mse = training_loss_mse / dataset_train.__len__()
        training_loss_mae = training_loss_mae / dataset_train.__len__()
            
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
                
                validation_loss_mse += criterion_mse(y_pred, y).item()
                validation_loss_mae += criterion_mae(y_pred, y).item()
                    
        validation_loss_mse = validation_loss_mse / dataset_validation.__len__()
        validation_loss_mae = validation_loss_mae / dataset_validation.__len__()
        
        if validation_loss_mse < best_validation_loss_mse:
            best_validation_loss_mse = validation_loss_mse
            counter_no_amelioration = 0
            best_model_state_dict = deepcopy(model.state_dict())
            best_epoch = epoch
            results = {
                '2. MSE training normalized': training_loss_mse,
                '2. MAE training normalized': training_loss_mae,
                '3. MSE validation normalized': validation_loss_mse,
                '3. MAE validation normalized': validation_loss_mae
            }
        else:
            counter_no_amelioration += 1
            if counter_no_amelioration == patience:
                model.load_state_dict(best_model_state_dict)
                break
                
    model.eval()
    
    test_loss_mse = 0
    test_loss_mae = 0
    test_loss_absolute_mse = 0
    test_loss_absolute_mae = 0
    error_absolute_max = 0
    error_relative_max = 0
    
    array_y_pred = np.array([])
    array_y = np.array([])
    array_other_target = np.array([])
    
    if target == 'y_bid':
        scaler_target = scaler_y_bid
        scaler_other_target = scaler_y_ask
    elif target == 'y_ask':
        scaler_target = scaler_y_ask
        scaler_other_target = scaler_y_bid
    else:
        raise NotImplementedError("Target not correct")
    
    for x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month, y_bid, y_ask in data_loader_test:
            
        if target == 'y_bid':
            y = y_bid
            array_other_target = np.concatenate([array_other_target, scaler_other_target.inverse_transform(y_ask.detach().cpu().unsqueeze(dim=1).numpy()).squeeze()])
        elif target == 'y_ask':
            y = y_ask
            array_other_target = np.concatenate([array_other_target, scaler_other_target.inverse_transform(y_bid.detach().cpu().unsqueeze(dim=1).numpy()).squeeze()])
        else:
            raise NotImplementedError("Target not correct")
                
        with torch.no_grad():
            
            y_pred = model.forward(x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month)
            
            test_loss_mse += criterion_mse(y_pred, y).item()
            test_loss_mae += criterion_mae(y_pred, y).item()
                        
            y_pred = scaler_target.inverse_transform(y_pred.detach().cpu().unsqueeze(dim=1).numpy()).squeeze()
            y = scaler_target.inverse_transform(y.detach().cpu().unsqueeze(dim=1).numpy()).squeeze()
            
            array_y_pred = np.concatenate([array_y_pred, y_pred])
            array_y = np.concatenate([array_y, y])
            
    results['4. MSE test normalized'] = test_loss_mse / dataset_test.__len__()
    results['4. MAE test normalized'] = test_loss_mae / dataset_test.__len__()
    results['5. MSE test absolute'] = mean_squared_error(array_y, array_y_pred)
    results['5. MAE test absolute'] = mean_absolute_error(array_y, array_y_pred)
    results['6. Error absolute max'] = np.max(np.abs(array_y - array_y_pred))
    results['6. Error relative max'] = np.max(np.abs((array_y - array_y_pred) / array_y))
    
    currency = 'EUR'
    sold = 1
    
    if target == 'y_bid':
        y_bid_t_1 = array_y_pred
        y_ask_t_1 = array_other_target
    elif target == 'y_ask':
        y_bid_t_1 = array_other_target
        y_ask_t_1 = array_y_pred
    else:
        raise NotImplementedError("Target not correct")
        
    y_bid_t_0 = scaler_y_bid.inverse_transform(dataset_test.X_now[:, 6].cpu().unsqueeze(dim=1).numpy()).squeeze()
    y_ask_t_0 = scaler_y_ask.inverse_transform(dataset_test.X_now[:, 2].cpu().unsqueeze(dim=1).numpy()).squeeze()
        
    for i in range(y_bid_t_0.shape[0]):
        
        if currency == "EUR":
            
            if y_bid_t_0[i] > y_ask_t_1[i]:
                
                currency = "CHF"
                sold = sold * y_bid_t_0[i]

        else:
            
            if y_ask_t_0[i] > y_bid_t_1[i]:
                
                currency = "EUR"
                sold = sold / y_ask_t_0[i]

    if currency == "CHF":
        
        sold = sold / y_ask_t_0[i]
                
    results['7. Financial performance'] = sold - 1
        
    return model, results, best_epoch