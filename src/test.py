import torch
import matplotlib.pyplot as plt
from data_loader import load_data
from utils import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(test_data, model):
    ground_1 = []
    ground_2 = []
    ground_3 = []
    pred_1 = []
    pred_2 = []
    pred_3 = []

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

    return ground_1, ground_2, ground_3, pred_1, pred_2, pred_3

def plot_results(loss_pts, train_err_pts, val_err_pts, ground_1, ground_2, ground_3, pred_1, pred_2, pred_3, fro=0, to=100):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(5, 1, figsize=(10, 10))

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
    axes[2].plot(ground_1[fro:to+1], label='Ground Truth', color='#3366FF')
    axes[2].plot(pred_1[fro:to+1], label='Prediction', color='orange')
    axes[2].grid(True)
    axes[2].legend(loc='upper right')

    axes[3].set_title(r'NO$_2$ - Nitrogen Dioxide')
    axes[3].set_xlabel('Timestamps (Randomized)')
    axes[3].set_ylabel(r'$\mu$g/m$^3$')
    axes[3].plot(ground_2[fro:to+1], label='Ground Truth', color='#3366FF')
    axes[3].plot(pred_2[fro:to+1], label='Prediction', color='orange')
    axes[3].grid(True)
    axes[3].legend(loc='upper right')

    axes[4].set_title(r'PT08.S5(O$_3$) - Indium Oxide')
    axes[4].set_xlabel('Timestamps (Randomized)')
    axes[4].set_ylabel('au')
    axes[4].plot(ground_3[fro:to+1], label='Ground Truth', color='#3366FF')
    axes[4].plot(pred_3[fro:to+1], label='Prediction', color='orange')
    axes[4].grid(True)
    axes[4].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def main():
    test_data, loss_pts, train_err_pts, val_err_pts = load_data()
    model = load_model()
    ground_1, ground_2, ground_3, pred_1, pred_2, pred_3 = test(test_data, model)
    plot_results(loss_pts, train_err_pts, val_err_pts, ground_1, ground_2, ground_3, pred_1, pred_2, pred_3)

if __name__ == "__main__":
    main()