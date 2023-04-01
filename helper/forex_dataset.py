import torch
from torch.utils.data import Dataset
import numpy as np


class ForexDataset(Dataset):
    
    def __init__(self, X_date, X_now, X_previous_hour, X_previous_day, X_previous_week, X_previous_month, y_bid, y_ask, idx, 
                 n_previous_hour_values, n_previous_day_values, n_previous_week_values, n_previous_month_values, n_features):
        
        self.X_date = X_date[idx].astype(np.int32)
        self.X_now = X_now[idx].astype(np.float32)
        self.X_previous_hour = X_previous_hour[idx].astype(np.float32)
        self.X_previous_day = X_previous_day[idx].astype(np.float32)
        self.X_previous_week = X_previous_week[idx].astype(np.float32)
        self.X_previous_month = X_previous_month[idx].astype(np.float32)
        self.y_bid = y_bid[idx].astype(np.float32)
        self.y_ask = y_ask[idx].astype(np.float32)
        self.n_previous_hour_values = n_previous_hour_values
        self.n_previous_day_values = n_previous_day_values
        self.n_previous_week_values = n_previous_week_values
        self.n_previous_month_values = n_previous_month_values
        self.n_features = n_features
        
    def __len__(self):
        
        return self.y_bid.shape[0]
    
    def __getitem__(self, idx):
        
        return self.X_date[idx], self.X_now[idx], self.X_previous_hour[idx], self.X_previous_day[idx], \
               self.X_previous_week[idx], self.X_previous_month[idx], self.y_bid[idx], self.y_ask[idx]
    
    def fit_scalers(self, scaler_now, scaler_previous_hour, scaler_previous_day, scaler_previous_week, scaler_previous_month, scaler_y_bid, scaler_y_ask):
        
        return scaler_now.fit(self.X_now), scaler_previous_hour.fit(self.X_previous_hour), scaler_previous_day.fit(self.X_previous_day), \
               scaler_previous_week.fit(self.X_previous_week), scaler_previous_month.fit(self.X_previous_month), \
               scaler_y_bid.fit(np.expand_dims(self.y_bid, axis=1)), scaler_y_ask.fit(np.expand_dims(self.y_ask, axis=1))
    
    def scale(self, scaler_now, scaler_previous_hour, scaler_previous_day, scaler_previous_week, scaler_previous_month, scaler_y_bid, scaler_y_ask):
        
        self.X_now = scaler_now.transform(self.X_now)
        self.X_previous_hour = scaler_previous_hour.transform(self.X_previous_hour).reshape(self.X_previous_hour.shape[0], self.n_previous_hour_values, self.n_features)
        self.X_previous_day = scaler_previous_day.transform(self.X_previous_day).reshape(self.X_previous_day.shape[0], self.n_previous_day_values, self.n_features)
        self.X_previous_week = scaler_previous_week.transform(self.X_previous_week).reshape(self.X_previous_week.shape[0], self.n_previous_week_values, self.n_features)
        self.X_previous_month = scaler_previous_month.transform(self.X_previous_month).reshape(self.X_previous_month.shape[0], self.n_previous_month_values, self.n_features)
        self.y_bid = np.squeeze(scaler_y_bid.transform(np.expand_dims(self.y_bid, axis=1)))
        self.y_ask = np.squeeze(scaler_y_ask.transform(np.expand_dims(self.y_ask, axis=1)))
        
    def transfer_to_tensor(self):
        
        self.X_date = torch.from_numpy(self.X_date)
        self.X_now = torch.from_numpy(self.X_now)
        self.X_previous_hour = torch.from_numpy(self.X_previous_hour)
        self.X_previous_day = torch.from_numpy(self.X_previous_day)
        self.X_previous_week = torch.from_numpy(self.X_previous_week)
        self.X_previous_month = torch.from_numpy(self.X_previous_month)
        self.y_bid = torch.from_numpy(self.y_bid)
        self.y_ask = torch.from_numpy(self.y_ask)
        
    def cuda(self):
        
        self.X_date = self.X_date.cuda()
        self.X_now = self.X_now.cuda()
        self.X_previous_hour = self.X_previous_hour.cuda()
        self.X_previous_day = self.X_previous_day.cuda()
        self.X_previous_week = self.X_previous_week.cuda()
        self.X_previous_month = self.X_previous_month.cuda()
        self.y_bid = self.y_bid.cuda()
        self.y_ask = self.y_ask.cuda()
        
    def cpu(self):
        
        self.X_date = self.X_date.cpu()
        self.X_now = self.X_now.cpu()
        self.X_previous_hour = self.X_previous_hour.cpu()
        self.X_previous_day = self.X_previous_day.cpu()
        self.X_previous_week = self.X_previous_week.cpu()
        self.X_previous_month = self.X_previous_month.cpu()
        self.y_bid = self.y_bid.cpu()
        self.y_ask = self.y_ask.cpu()