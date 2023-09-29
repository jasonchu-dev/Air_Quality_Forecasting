import torch
import yaml
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loader import load_and_preprocess_data, save_data
from utils import print_get_results, get_path, save_model
from model import CRNN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = get_path('../models', file_offset=1)

    with open('../configs/hyperparameters.yaml', 'r') as file:
        hyperparameters = yaml.safe_load(file)

    lr = hyperparameters['lr']
    weight_decay = hyperparameters['weight_decay']
    dropout = hyperparameters['dropout']
    factor = hyperparameters['factor']
    patience = hyperparameters['patience']
    min_lr = hyperparameters['min_lr']
    train_size = hyperparameters['train_size']
    batch_size = hyperparameters['batch_size']
    epochs = hyperparameters['epochs']
    reduction = hyperparameters['reduction']
    mode = hyperparameters['mode']

    model = CRNN(input_dim=3, hidden_dim=24, layer_dim=2, output_dim=3, dropout=dropout).to(device)
    criterion = nn.MSELoss(reduction=reduction)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=True, min_lr=min_lr)

    loss_pts = []
    train_err_pts = []
    val_err_pts = []
    lowest_val_err = float('inf')

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
                save_model(model_path, model, optimizer, loss, best_epoch)

        scheduler.step(loss)
        val_avg_err = float(torch.sum(val_avg_err)/len(val_avg_err))

        train_loss, train_avg_err, val_avg_err = print_get_results(epoch, loss.item(), train_avg_err, val_avg_err)
        loss_pts.append(train_loss)
        train_err_pts.append(train_avg_err)
        val_err_pts.append(val_avg_err)

    save_data(test_data, loss_pts, train_err_pts, val_err_pts)

if __name__ == "__main__":
    main()