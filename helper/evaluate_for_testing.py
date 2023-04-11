import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc


def evaluate_financial_performace(array_y_bid_t_0, array_y_bid_t_1, array_y_ask_t_0, array_y_ask_t_1):
    
    currency = 'EUR'
    sold = 1.0
              
    for y_bid_t_0, y_bid_t_1, y_ask_t_0, y_ask_t_1 in zip(array_y_bid_t_0, array_y_bid_t_1, array_y_ask_t_0, array_y_ask_t_1):
            
        if currency == 'EUR':
            if y_bid_t_0 > y_ask_t_1:
                currency = 'CHF'
                sold = sold * y_bid_t_0
        else:
            if y_ask_t_0 < y_bid_t_1:
                currency = 'EUR'
                sold = sold / y_ask_t_0
                    
    if currency == 'CHF':
        sold = sold / y_ask_t_0
        
    return sold - 1


def evaluate_for_testing(model, dataset_train, dataset_validation, dataset_test, scaler_y_bid, scaler_y_ask, optimizer, batch_size_train, batch_size_validation, batch_size_test, learning_rate, weight_decay, patience, epochs):
    
    # Set random seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Transfer model and datasets to GPU
    model.cuda()
    dataset_train.cuda()
    dataset_validation.cuda()
    dataset_test.cuda()
    
    # Loss functions
    criterion_mse_y_bid = nn.MSELoss(reduction='sum')
    criterion_mae_y_bid = nn.L1Loss(reduction='sum')
    criterion_mse_y_ask = nn.MSELoss(reduction='sum')
    criterion_mae_y_ask = nn.L1Loss(reduction='sum')

    # Optimizer
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError("Optimizer {} not implemented".format(optimizer))
    
    # Wrap datasets into dataloader
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)
    data_loader_validation = DataLoader(dataset_validation, batch_size=batch_size_validation, shuffle=False)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False)
    
    # Early stopping variables
    best_validation_loss_mse = np.inf
    best_epoch = 0
    counter_no_amelioration = 0
    best_model_state_dict = deepcopy(model.state_dict())
    
    results = dict()
    
     # Training/validation loop
    for epoch in range(1, epochs + 1):
        
        # Training
        model = model.train()
        training_loss_mse_y_bid, training_loss_mae_y_bid, training_loss_mse_y_ask, training_loss_mae_y_ask = 0, 0, 0, 0
        
        # Iterate over train dataset
        for x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month, y_bid, y_ask in data_loader_train:
                
            # Set the gradient to none before each iteration
            optimizer.zero_grad(set_to_none=True)
                                        
            # Forward pass
            y_bid_pred, y_ask_pred = model.forward(x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month)
            
            # Compute losses
            loss_mse_y_bid = criterion_mse_y_bid(y_bid_pred, y_bid)
            loss_mae_y_bid = criterion_mae_y_bid(y_bid_pred, y_bid)
            loss_mse_y_ask = criterion_mse_y_ask(y_ask_pred, y_ask)
            loss_mae_y_ask = criterion_mae_y_ask(y_ask_pred, y_ask)
            
            training_loss_mse_y_bid += loss_mse_y_bid.item()
            training_loss_mae_y_bid += loss_mae_y_bid.item()
            training_loss_mse_y_ask += loss_mse_y_ask.item()
            training_loss_mae_y_ask += loss_mae_y_ask.item()
            
            # Backward propagation on mse loss
            (loss_mse_y_bid + loss_mse_y_ask).backward()
            
            # Update optimizer step
            optimizer.step()
            
        # Divide training losses by dataset len
        training_loss_mse_y_bid = training_loss_mse_y_bid / dataset_train.__len__()
        training_loss_mae_y_bid = training_loss_mae_y_bid / dataset_train.__len__()
        training_loss_mse_y_ask = training_loss_mse_y_ask / dataset_train.__len__()
        training_loss_mae_y_ask = training_loss_mae_y_ask / dataset_train.__len__()
            
        # Validation
        model.eval()
        validation_loss_mse_y_bid, validation_loss_mae_y_bid, validation_loss_mse_y_ask, validation_loss_mae_y_ask = 0, 0, 0, 0
        
        # Iterate over validation dataset
        for x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month, y_bid, y_ask in data_loader_validation:
                
            # Disable gradient during validation
            with torch.no_grad():
                
                # Forward pass
                y_bid_pred, y_ask_pred = model.forward(x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month)
                
                # Compute losses
                loss_mse_y_bid = criterion_mse_y_bid(y_bid_pred, y_bid)
                loss_mae_y_bid = criterion_mae_y_bid(y_bid_pred, y_bid)
                loss_mse_y_ask = criterion_mse_y_ask(y_ask_pred, y_ask)
                loss_mae_y_ask = criterion_mae_y_ask(y_ask_pred, y_ask)

                validation_loss_mse_y_bid += loss_mse_y_bid.item()
                validation_loss_mae_y_bid += loss_mae_y_bid.item()
                validation_loss_mse_y_ask += loss_mse_y_ask.item()
                validation_loss_mae_y_ask += loss_mae_y_ask.item()
                    
        # Divide validation losses by dataset len
        validation_loss_mse_y_bid = validation_loss_mse_y_bid / dataset_validation.__len__()
        validation_loss_mae_y_bid = validation_loss_mae_y_bid / dataset_validation.__len__()
        validation_loss_mse_y_ask = validation_loss_mse_y_ask / dataset_validation.__len__()
        validation_loss_mae_y_ask = validation_loss_mae_y_ask / dataset_validation.__len__()
        
        validation_loss_mse = validation_loss_mse_y_bid + validation_loss_mse_y_ask
        
        # Earlystopping management
        if validation_loss_mse < best_validation_loss_mse:
            # If validation loss is better than the best validation loss achieved, update best validation loss
            # and reset the counter then stock the losses values and best epoch
            best_validation_loss_mse = validation_loss_mse
            counter_no_amelioration = 0
            best_model_state_dict = deepcopy(model.state_dict())
            best_epoch = epoch
            results['2. MSE training y_bid normalized'] = training_loss_mse_y_bid
            results['2. MAE training y_bid normalized'] = training_loss_mae_y_bid
            results['3. MSE validation y_bid normalized'] = validation_loss_mse_y_bid
            results['3. MAE validation y_bid normalized'] = validation_loss_mae_y_bid
            results['2. MSE training y_ask normalized'] = training_loss_mse_y_ask
            results['2. MAE training y_ask normalized'] = training_loss_mae_y_ask
            results['3. MSE validation y_ask normalized'] = validation_loss_mse_y_ask
            results['3. MAE validation y_ask normalized'] = validation_loss_mae_y_ask
        else:
            # If not, increase the counter. If the counter is equal to the limit, stop training/validation loop
            counter_no_amelioration += 1
            if counter_no_amelioration == patience:
                model.load_state_dict(best_model_state_dict)
                break
    
    # Test
    model.eval()
    list_y_bid, list_y_ask = [], []
    list_y_bid_pred, list_y_ask_pred = [], []
    
    # Iterate over test dataset
    for x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month, y_bid, y_ask in data_loader_test:
                
        # Disable gradient during test
        with torch.no_grad():
            
            # Forward pass
            y_bid_pred, y_ask_pred = model.forward(x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month)
            
            # Stock y_bid, y_ask, y_bid_pred and y_ask_pred
            list_y_bid.append(y_bid.detach().cpu().numpy())
            list_y_ask.append(y_ask.detach().cpu().numpy())
            list_y_bid_pred.append(y_bid_pred.detach().cpu().numpy())
            list_y_ask_pred.append(y_ask_pred.detach().cpu().numpy())
            
    # Concatenate arrays
    array_y_bid = np.concatenate(list_y_bid)
    array_y_ask = np.concatenate(list_y_ask)
    array_y_bid_pred = np.concatenate(list_y_bid_pred)
    array_y_ask_pred = np.concatenate(list_y_ask_pred)
    
    # Unscale values
    array_y_bid_unscaled = scaler_y_bid.inverse_transform(np.expand_dims(array_y_bid, axis=1)).squeeze()
    array_y_ask_unscaled = scaler_y_ask.inverse_transform(np.expand_dims(array_y_ask, axis=1)).squeeze()
    array_y_bid_pred_unscaled = scaler_y_bid.inverse_transform(np.expand_dims(array_y_bid_pred, axis=1)).squeeze()
    array_y_ask_pred_unscaled = scaler_y_ask.inverse_transform(np.expand_dims(array_y_ask_pred, axis=1)).squeeze()

    # Compute losses and other metrics
    results['4. MSE test y_bid normalized'] = mean_squared_error(array_y_bid, array_y_bid_pred)
    results['4. MAE test y_bid normalized'] = mean_absolute_error(array_y_bid, array_y_bid_pred)
    results['5. MSE test y_bid absolute'] = mean_squared_error(array_y_bid_unscaled, array_y_bid_pred_unscaled)
    results['5. MAE test y_bid absolute'] = mean_absolute_error(array_y_bid_unscaled, array_y_bid_pred_unscaled)
    results['6. Error max absolute y_bid'] = np.max(np.abs(array_y_bid_unscaled - array_y_bid_pred_unscaled))
    results['6. Error max relative y_bid'] = np.max(np.abs((array_y_bid_unscaled - array_y_bid_pred_unscaled) / array_y_bid_unscaled))
    
    results['4. MSE test y_ask normalized'] = mean_squared_error(array_y_ask, array_y_ask_pred)
    results['4. MAE test y_ask normalized'] = mean_absolute_error(array_y_ask, array_y_ask_pred)
    results['5. MSE test y_ask absolute'] = mean_squared_error(array_y_ask_unscaled, array_y_ask_pred_unscaled)
    results['5. MAE test y_ask absolute'] = mean_absolute_error(array_y_ask_unscaled, array_y_ask_pred_unscaled)
    results['6. Error max absolute y_ask'] = np.max(np.abs(array_y_ask_unscaled - array_y_ask_pred_unscaled))
    results['6. Error max relative y_ask'] = np.max(np.abs((array_y_ask_unscaled - array_y_ask_pred_unscaled) / array_y_ask_unscaled))
        
    # Compute financial performance metric
    array_y_bid_t_0 = array_y_bid_unscaled[:-1]
    array_y_bid_t_1 = array_y_bid_pred_unscaled[1:]
    array_y_ask_t_0 = array_y_ask_unscaled[:-1]
    array_y_ask_t_1 = array_y_ask_unscaled[1:]        
    results['7. Financial performance y_bid'] = evaluate_financial_performace(array_y_bid_t_0, array_y_bid_t_1, array_y_ask_t_0, array_y_ask_t_1)
        
    array_y_bid_t_0 = array_y_bid_unscaled[:-1]
    array_y_bid_t_1 = array_y_bid_unscaled[1:]
    array_y_ask_t_0 = array_y_ask_unscaled[:-1]
    array_y_ask_t_1 = array_y_ask_pred_unscaled[1:] 
    results['7. Financial performance y_ask'] = evaluate_financial_performace(array_y_bid_t_0, array_y_bid_t_1, array_y_ask_t_0, array_y_ask_t_1)

    array_y_bid_t_0 = array_y_bid_unscaled[:-1]
    array_y_bid_t_1 = array_y_bid_pred_unscaled[1:]
    array_y_ask_t_0 = array_y_ask_unscaled[:-1]
    array_y_ask_t_1 = array_y_ask_pred_unscaled[1:] 
    results['7. Financial performance'] = evaluate_financial_performace(array_y_bid_t_0, array_y_bid_t_1, array_y_ask_t_0, array_y_ask_t_1)
        
    # Return model and datasets to CPU and empty cache
    model.cpu()
    dataset_train.cpu()
    dataset_validation.cpu()
    dataset_test.cpu()
    gc.collect()
    torch.cuda.empty_cache()
        
    return model, results, best_epoch