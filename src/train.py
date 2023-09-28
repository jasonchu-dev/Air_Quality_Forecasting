import os
import h5py
import torch
import pickle
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loader import load_and_preprocess_data
from utils import save_checkpoint, format_number
from model import CRNN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs('../models', exist_ok=True)
    file_list = os.listdir('../models')
    num_files = len(file_list)
    file_path = f'../models/model_{num_files + 1}.tar'

    lr = 1e-2
    weight_decay = 0
    dropout = 0

    model = CRNN(input_dim=3, hidden_dim=24, layer_dim=2, output_dim=3, dropout=dropout).to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=25, verbose=True, min_lr=1e-8)

    epochs = 100

    loss_pts = []
    train_err_pts = []
    val_err_pts = []
    lowest_val_err = float('inf')

    batch_size = 300
    train_size = 0.8

    train_data, val_data, test_data = load_and_preprocess_data(batch_size=batch_size, train_size=train_size)

    for epoch in range(1, epochs+1):
        train_avg_err = torch.tensor([]).to(device)
        model.train()
        for feature, label in train_data:
            optimizer.zero_grad()
            feature = feature.transpose(2, 1).to(device)
            y_pred = model(feature)
            y_pred = y_pred.transpose(2, 1)
            loss = criterion(y_pred, label.to(device))
            loss.backward()
            optimizer.step()

            train_avg_err = torch.cat((torch.reshape((torch.sum(abs(label.to(device) - y_pred))/(batch_size*3)),(-1,)),train_avg_err), dim=0).to(device)
        
        train_avg_err = float(torch.sum(train_avg_err)/len(train_avg_err))

        val_avg_err = torch.tensor([]).to(device)
        model.eval()
        with torch.no_grad():
            for feature, label in val_data:
                feature = feature.transpose(2, 1).to(device)
                y_pred = model(feature)
                y_pred = y_pred.transpose(2, 1)

                val_avg_err = torch.cat((torch.reshape((torch.sum(abs(label.to(device) - y_pred))/(batch_size*3)),(-1,)),val_avg_err), dim=0).to(device)

            if lowest_val_err > float(torch.sum(val_avg_err)/len(val_avg_err)):
                lowest_val_err = float(torch.sum(val_avg_err)/len(val_avg_err))
                best_epoch = epoch
                save_checkpoint(file_path, model, optimizer, loss, best_epoch)

        scheduler.step(loss)
        val_avg_err = float(torch.sum(val_avg_err)/len(val_avg_err))

        epoch = str(epoch).rjust(3)
        train_loss = format_number(loss.item())
        train_avg_err = format_number(train_avg_err)
        val_avg_err = format_number(val_avg_err)

        print(f'Epoch: {epoch} | Loss: {train_loss} | Train Avg Err: {train_avg_err} | val Avg Err: {val_avg_err}')
        loss_pts.append(loss.item())
        train_err_pts.append(float(train_avg_err))
        val_err_pts.append(float(val_avg_err))

    file_list = os.listdir('../data')
    num_files = len(file_list)
    file_path = f'../data/test_data_{num_files}.pkl'

    with open(file_path, 'wb') as file:
        pickle.dump(test_data, file)

    os.makedirs('../logs', exist_ok=True)
    file_list = os.listdir('../logs')
    num_files = len(file_list)
    file_path = f'../logs/run_{num_files + 1}.h5'

    with h5py.File(file_path, 'w') as f:
        f.create_dataset('loss_pts', data=loss_pts)
        f.create_dataset('train_err_pts', data=train_err_pts)
        f.create_dataset('val_err_pts', data=val_err_pts)

if __name__ == "__main__":
    main()