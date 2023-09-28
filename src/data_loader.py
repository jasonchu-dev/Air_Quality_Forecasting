import torch
import dask
import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

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

    # -200 are nans

    for i in df.columns[2:]:
        df[i] = df[i].replace(-200, df[i].mean())

    # 2004-03-10
    # Timestamp('2004-03-10 00:00:00')
    # call date() on object to get 'datetime.date(2004, 3, 10)'
    # attributes are year, month, day, weekday()

    df['Year'] = df['Date'].map(lambda x: x.date().year - 2004, meta=('Date', 'i8'))
    df['Month'] = df['Date'].map(lambda x: x.date().month, meta=('Date', 'i8'))
    df['Weekday'] = df['Date'].map(lambda x: x.date().weekday(), meta=('Date', 'i8'))

    # 18:00:00
    # datetime.time(18, 0)
    # attributes are hour, minute, second, microsecond
    #            [0, 23], [0, 59], [0, 59], [0, 999999]

    if type(df['Time'].compute()[0]) == str:
        df['Hour'] = df['Time'].map(lambda x: int(x[:2]), meta=('Time', 'i8'))
    else:
        df['Hour'] = df['Time'].map(lambda x: x.hour, meta=('Time', 'i8'))

    del df['Date']
    del df['Time']

    columns = list(df.columns)
    first_four_columns = columns[-4:]
    remaining_columns = columns[:-4]

    new_order = first_four_columns + remaining_columns
    df = df[new_order]

    scaler_pipeline = Pipeline([
        ('robust_scaler', RobustScaler()),
        ('standard_scaler', StandardScaler()),
        ('min_max_scaler', MinMaxScaler())
    ])

    cols = list(df.columns)
    scaled_data = scaler_pipeline.fit_transform(df)
    df = dd.from_pandas(pd.DataFrame(scaled_data, columns=cols), npartitions=2)

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