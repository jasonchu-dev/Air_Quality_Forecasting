import h5py
import pickle
import torch
import dask
import dask.dataframe as dd
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils import feature_engr, reorder, norm, get_path

class CustomDataset(Dataset):
    def __init__(self, features, labels, dtype=torch.float):
        self.features = torch.tensor(features, dtype=dtype)
        self.labels = torch.tensor(labels, dtype=dtype)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

def load_and_preprocess_data(batch_size=300, train_size=0.8):
    parts = dask.delayed(pd.read_excel)('../data/AirQualityUCI.xlsx')
    df = dd.from_delayed(parts)

    df = feature_engr(df)
    df = reorder(df)
    df = norm(df)

    scaled_df = df.drop(columns=['CO(GT)', 'NMHC(GT)', 'AH', 'NOx(GT)', 'PT08.S4(NO2)', 'T', 'RH'])

    X = np.array(scaled_df.iloc[:, :7])
    y = np.array(scaled_df.iloc[:, 7:])

    X = X.reshape(3119, 3, 7)
    y = y.reshape(3119, 3, 3)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, train_size=train_size, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, shuffle=False)

    train_data = torch.utils.data.DataLoader(
        CustomDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False
    )

    val_data = torch.utils.data.DataLoader(
        CustomDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False
    )

    test_data = torch.utils.data.DataLoader(
        CustomDataset(X_test, y_test, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False
    )

    return train_data, val_data, test_data

def save_data(test_data, loss_pts, train_err_pts, val_err_pts):
    file_path = get_path('../data', file_offset=0)
    with open(file_path, 'wb') as file:
        pickle.dump(test_data, file)

    file_path = get_path('../logs', file_offset=1)
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('loss_pts', data=loss_pts)
        f.create_dataset('train_err_pts', data=train_err_pts)
        f.create_dataset('val_err_pts', data=val_err_pts)

def load_data():
    file_path = get_path('../data', file_offset=-1)
    with open(file_path, 'rb') as file:
        test_data = pickle.load(file)
        
    file_path = get_path('../logs', file_offset=0)
    with h5py.File(file_path, 'r') as f:
        loss_pts = np.array(f['loss_pts'])
        train_err_pts = np.array(f['train_err_pts'])
        val_err_pts = np.array(f['val_err_pts'])

    return test_data, loss_pts, train_err_pts, val_err_pts