import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
import gc


def evaluate_for_tuning(model, dataset_train, dataset_validation, target, optimizer, batch_size_train, batch_size_validation, learning_rate, weight_decay, patience, epochs, flag_transfer_cpu_gpu):
    
    # Set random seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Transfer model and datasets to GPU
    if flag_transfer_cpu_gpu == True:
        dataset_train.cuda()
        dataset_validation.cuda()
        
    model.cuda()

    # Loss functions
    criterion_mse_y_bid = nn.MSELoss(reduction='sum')
    criterion_mse_y_ask = nn.MSELoss(reduction='sum')

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
    
    # Early stopping variables
    best_validation_loss = np.inf
    counter_no_amelioration = 0
    
    # Training/validation loop
    for epoch in range(epochs):
        
        # Training
        model = model.train()
        
        # Iterate over train dataset
        for x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month, y_bid, y_ask in data_loader_train:
                
            # Set the gradient to none before each iteration
            optimizer.zero_grad(set_to_none=True)
                         
            # Forward pass
            y_bid_pred, y_ask_pred = model.forward(x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month)
            
            # Compute loss
            if target == 'y_bid':
                loss = criterion_mse_y_bid(y_bid_pred, y_bid)
            elif target == 'y_ask':
                loss = criterion_mse_y_ask(y_ask_pred, y_ask)
            elif target == 'dual':
                loss = criterion_mse_y_bid(y_bid_pred, y_bid) + criterion_mse_y_ask(y_ask_pred, y_ask)
            else:
                raise NotImplementedError("Target {} not implemented".format(target))
            
            # Backward propagation
            loss.backward()
            
            # Update optimizer step
            optimizer.step()
            
        # Validation
        model.eval()
        validation_loss = 0
        
        # Iterate over validation dataset
        for x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month, y_bid, y_ask in data_loader_validation:
                
            # Disable gradient during validation
            with torch.no_grad():
                
                # Forward pass
                y_bid_pred, y_ask_pred = model.forward(x_date, x_now, x_previous_hour, x_previous_day, x_previous_week, x_previous_month)
                
                # Compute loss
                if target == 'y_bid':
                    loss = criterion_mse_y_bid(y_bid_pred, y_bid)
                elif target == 'y_ask':
                    loss = criterion_mse_y_ask(y_ask_pred, y_ask)
                elif target == 'dual':
                    loss = criterion_mse_y_bid(y_bid_pred, y_bid) + criterion_mse_y_ask(y_ask_pred, y_ask)
                else:
                    raise NotImplementedError("Target {} not implemented".format(target))
                
                validation_loss += loss.item()
                    
        # Divide validation loss by dataset len
        validation_loss = validation_loss / dataset_validation.__len__()
                
        # Earlystopping management
        if validation_loss < best_validation_loss:
            # If validation loss is better than the best validation loss achieved, update best validation loss
            # and reset the counter
            best_validation_loss = validation_loss
            counter_no_amelioration = 0
        else:
            # If not, increase the counter. If the counter is equal to the limit, stop training/validation loop
            counter_no_amelioration += 1
            if counter_no_amelioration == patience:
                break
              
    # Return model and datasets to CPU and empty cache
    if flag_transfer_cpu_gpu == True:
        dataset_train.cpu()
        dataset_validation.cpu()
        
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
                
    return best_validation_loss