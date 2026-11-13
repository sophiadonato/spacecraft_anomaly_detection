import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

# Dataset loader creation : training_ds only (unsupervised technique) BUT the last part of the training_ds is used for validation 
# training_ds = X_train, y_train with
# - X_train = sliding windows of the last 250 values (telemetry + commands)
# - y_train = true value of the telemetry after each window
def ds_creation(channel_id, window_size=250):
    channel_values = np.load(f"../data/train/{channel_id}.npy") # (n_timesteps, n_inputs)
    X = np.stack([channel_values[i:i+window_size, :] for i in range(len(channel_values)-window_size)]) # (n_timesteps-250 = n_windows, window_size, n_inputs)
    y  = channel_values[window_size : , 0].reshape(-1,1) # (n_windows, 1)
    return X, y

def ds_loaders(X, y, validation_split=0.2, batch_size=64):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    n_val = int(validation_split*len(X_t))
    X_train_t, X_val_t = X_t[:-n_val], X_t[-n_val:]
    y_train_t, y_val_t = y_t[:-n_val], y_t[-n_val:]

    training_dataset = TensorDataset(X_train_t, y_train_t)
    validation_dataset = TensorDataset(X_val_t, y_val_t)
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return training_loader, validation_loader

# Network : (reminder : 1 network per channel ie per file so create class)
class telemetryPredictorLSTM(nn.Module):
    def __init__(self, input_size, layers = [80, 80], dropout = 0.3, n_predictions = 1): 
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, layers[0], batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(layers[0], layers[1], batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(layers[1], n_predictions)
    def forward(self, x):
        out, _ = self.lstm1(x) # nn.LSTM outputs (output, (h_n, c_n)) -> keep only output
        out = self.drop1(out)

        out, _ = self.lstm2(out) 
        out = out[:,-1,:] # Selecting the last time step
        out = self.drop2(out)

        out = self.fc(out)

        return out

# Training fct
def train(num_epochs, training_loader, validation_loader, network, criterion, optimizer, device):
    train_avg_losses = []
    val_avg_losses = []
    for i in range(num_epochs): 
        print(f"Epoch number {i+1} starting : ")
        train_losses = []
        val_losses = []

        network.train()
        for x, y in training_loader : 
            x,y = x.to(device), y.to(device)
            y_pred = network(x)

            optimizer.zero_grad()
            loss = criterion(y_pred, y)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Val once per epoch
        network.eval()
        val_losses = []
        with torch.no_grad() :
            for x, y in validation_loader : 
                x, y = x.to(device), y.to(device)
                y_pred = network(x)
                loss = criterion(y_pred, y)
                val_losses.append(loss.item())

        # Compute avg loss and store them
        train_avg_loss = np.mean(train_losses)
        val_avg_loss = np.mean(val_losses)
        print(f"Epoch {i+1} : train loss = {train_avg_loss}, vall loss = {val_avg_loss}")

        train_avg_losses.append(train_avg_loss)
        val_avg_losses.append(val_avg_loss)

    return train_avg_losses, val_avg_losses


# Training loop for a single file :
chan_name = "A-1"

# GPU 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
device = torch.device(device)

# Input size 
INPUT_SIZES = {"SMAP": 25, "MSL": 55}
labels = pd.read_csv("../data/labeled_anomalies.csv")
spacecraft_name = labels.loc[labels.chan_id == chan_name, "spacecraft"].iloc[0]
input_size = INPUT_SIZES[spacecraft_name]

# Net init 
net = telemetryPredictorLSTM(input_size=input_size)
net.to(device)

# Loss
criterion = nn.MSELoss()

# Optimizer 
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# Data loaders
X, y = ds_creation(chan_name)
training_loader, validation_loader = ds_loaders(X, y)

# Train
train_avg_losses, val_avg_losses = train(35, training_loader, validation_loader, net, criterion, optimizer, device)