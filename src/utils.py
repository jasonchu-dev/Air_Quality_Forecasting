import os
import torch
import dask.dataframe as dd
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from model import CRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def feature_engr(df):
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
    return df

def reorder(df):
    columns = list(df.columns)
    first_four_columns = columns[-4:]
    remaining_columns = columns[:-4]
    new_order = first_four_columns + remaining_columns
    return df[new_order]

def norm(df):
    scaler_pipeline = Pipeline([
        ('robust_scaler', RobustScaler()),
        ('standard_scaler', StandardScaler()),
        ('min_max_scaler', MinMaxScaler())
    ])

    cols = list(df.columns)
    scaled_data = scaler_pipeline.fit_transform(df)
    df = dd.from_pandas(pd.DataFrame(scaled_data, columns=cols), npartitions=2)
    return df

def format_number(num, length=20):
    num_str = str(num)

    if "." in num_str:
        integer_part, decimal_part = num_str.split(".")
        formatted_decimal_part = decimal_part.ljust(3, '0')
        formatted_integer_part = integer_part
        formatted_num = "{}.{}".format(formatted_integer_part, formatted_decimal_part)
    else:
        formatted_num = num_str

    formatted_num = formatted_num.ljust(length, '0')
    return formatted_num

def print_get_results(epoch, loss, train_avg_err, val_avg_err):
    epoch = str(epoch).rjust(3)
    train_loss = format_number(loss)
    train_avg_err = format_number(train_avg_err)
    val_avg_err = format_number(val_avg_err)

    print(f'Epoch: {epoch} | Loss: {train_loss} | Train Avg Err: {train_avg_err} | val Avg Err: {val_avg_err}')

    return loss, float(train_avg_err), float(val_avg_err)

def get_path(dir, file_offset=0):
    os.makedirs(dir, exist_ok=True)
    file_list = os.listdir(dir)
    num_files = len(file_list)
    num_files += file_offset

    if dir == '../models':
        file_path = f'{dir}/model_{num_files}.tar'
    elif dir == '../data':
        file_path = f'{dir}/test_data_{num_files}.pkl'
    else:
        file_path = f'{dir}/run_{num_files}.h5'

    return file_path

def save_model(file_path, model, optimizer, loss, epoch):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, file_path)

def load_model():
    file_list = os.listdir('../models')
    num_files = len(file_list)
    file_path = f'../models/model_{num_files}.tar'

    model = CRNN().to(device)
    checkpoint = torch.load(file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])

    return model