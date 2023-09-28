import torch
import torch.nn as nn
import torch.nn.init as init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EB(nn.Module):
    """Expansion Block"""
    def __init__(self, input_dim, output_dim, kernel_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size),
            nn.BatchNorm1d(output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        return self.model(input)
    
class CB(nn.Module):
    """Compression Block"""
    def __init__(self, input_dim, output_dim, kernel_size, stride):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size, stride),
            nn.BatchNorm1d(output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        return self.model(input)

class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network"""
    def __init__(self, input_dim=3, hidden_dim=24, layer_dim=2, output_dim=3, dropout=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.expansion_cnn = nn.Sequential(
            EB(7, 28, 1),
            EB(28, 84, 1)
        )
        self.rnn = nn.LSTM(input_size=input_dim, num_layers=layer_dim, hidden_size=hidden_dim, batch_first=True, dropout=dropout)
        self.compression_cnn = nn.Sequential(
            CB(84, 42, 6, 2),
            CB(42, 12, 4, 3)
        )
        self.conv1d = nn.Conv1d(12, output_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        for layer in [self.expansion_cnn, self.rnn, self.compression_cnn, self.conv1d]:
            if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
    
    def forward(self, input):
        h0, c0 = self.init_hidden(input)
        output = self.expansion_cnn(input)
        output, (hn, cn) = self.rnn(output, (h0, c0))
        output = self.compression_cnn(output)
        output = self.conv1d(output)
        output = self.sigmoid(output)
        return output
    
    def init_hidden(self, input):
        h0 = torch.zeros(self.layer_dim, input.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, input.size(0), self.hidden_dim).to(device)
        return [t for t in (h0, c0)]