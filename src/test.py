import os
import h5py
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from model import CRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.style.use('dark_background')

ground_1 = []
ground_2 = []
ground_3 = []
pred_1 = []
pred_2 = []
pred_3 = []

file_list = os.listdir('../data')
num_files = len(file_list)
file_path = f'../data/test_data_{num_files - 1}.pkl'

with open(file_path, 'rb') as file:
    test_data = pickle.load(file)

file_list = os.listdir('../logs')
num_files = len(file_list)
file_path = f'../logs/run_{num_files}.h5'

with h5py.File(file_path, 'r') as f:
    loss_pts = np.array(f['loss_pts'])
    train_err_pts = np.array(f['train_err_pts'])
    val_err_pts = np.array(f['val_err_pts'])

file_list = os.listdir('../models')
num_files = len(file_list)
file_path = f'../models/model_{num_files}.tar'

model = CRNN().to(device)
checkpoint = torch.load(file_path, map_location=device)
model.load_state_dict(checkpoint['model_state'])

with torch.no_grad():
    for feature, label in test_data:
        ground_1.extend(label[:, :, 0].reshape(-1).tolist())
        ground_2.extend(label[:, :, 1].reshape(-1).tolist())
        ground_3.extend(label[:, :, 2].reshape(-1).tolist())
        feature = feature.transpose(2, 1).to(device)
        y_pred = model(feature)
        y_pred = y_pred.transpose(2, 1)
        pred_1.extend(y_pred[:, :, 0].reshape(-1).tolist())
        pred_2.extend(y_pred[:, :, 1].reshape(-1).tolist())
        pred_3.extend(y_pred[:, :, 2].reshape(-1).tolist())
    
fig, axes = plt.subplots(5, 1, figsize=(10, 10))

i = 0

axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].plot(loss_pts, label='MSE', color='#93FF33')
axes[0].grid(True)
axes[0].legend(loc='upper right')

axes[1].set_title('Avg Err')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Err')
axes[1].plot(train_err_pts, label='Train Err', color='#9933FF')
axes[1].plot(val_err_pts, label='Val Err', color='#FF0000')
axes[1].grid(True)
axes[1].legend(loc='upper right')

axes[2].set_title(r'PT08.S3(NO$_x$) - Tungsten Oxide')
axes[2].set_xlabel('Timestamps (Randomized)')
axes[2].set_ylabel('au')
axes[2].plot(ground_1[i:i+100], label='Ground Truth', color='#3366FF')
axes[2].plot(pred_1[i:i+100], label='Prediction', color='orange')
axes[2].grid(True)
axes[2].legend(loc='upper right')

axes[3].set_title(r'NO$_2$ - Nitrogen Dioxide')
axes[3].set_xlabel('Timestamps (Randomized)')
axes[3].set_ylabel(r'$\mu$g/m$^3$')
axes[3].plot(ground_2[i:i+100], label='Ground Truth', color='#3366FF')
axes[3].plot(pred_2[i:i+100], label='Prediction', color='orange')
axes[3].grid(True)
axes[3].legend(loc='upper right')

axes[4].set_title(r'PT08.S5(O$_3$) - Indium Oxide')
axes[4].set_xlabel('Timestamps (Randomized)')
axes[4].set_ylabel('au')
axes[4].plot(ground_3[i:i+100], label='Ground Truth', color='#3366FF')
axes[4].plot(pred_3[i:i+100], label='Prediction', color='orange')
axes[4].grid(True)
axes[4].legend(loc='upper right')

plt.tight_layout()
plt.show()