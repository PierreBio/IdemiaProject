import datetime
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
import ast

def csv_string_to_list(string):
    try:
        return ast.literal_eval(string)
    except ValueError:
        raise Exception("Could not process data, verify csv file")

def record_hyperparameters_performance(lr, batch_size, rmse, layers, activation_fn, epochs, csv_file="./results/model_performance.csv"):
    results_dir = os.path.dirname(csv_file)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    data = {
        "Date": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Learning Rate": [lr],
        "Batch Size": [batch_size],
        "Layers": [str(layers)],
        "Activation Function": [activation_fn],
        "RMSE": [rmse],
        "Epochs": [epochs]
    }
    df = pd.DataFrame(data)
    if os.path.isfile(csv_file):
        existing_df = pd.read_csv(csv_file, sep=";")
        df = pd.concat([existing_df, df])
    df.sort_values(by="RMSE", inplace=True)
    df.to_csv(csv_file, index=False, header=True, sep=";")

# Define the Dataset Class
class KeypointsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Define the Neural Network Model
class MLP(nn.Module):
    def __init__(self, input_size, output_size, layers, activation_fn):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()

        # Créer les couches cachées dynamiquement
        last_size = input_size
        for i, layer_size in enumerate(layers):
            self.layers.add_module(f"Linear_{i}", nn.Linear(last_size, layer_size))
            self.layers.add_module(f"Activation_{i}", activation_fn())
            last_size = layer_size

        # Ajouter la dernière couche
        self.layers.add_module("Output", nn.Linear(last_size, output_size))

    def forward(self, x):
        return self.layers(x)


# Reading Train CSV data
df_train = pd.read_csv('train_data_original.csv')
df_train['keypoints'] = df_train['keypoints'].apply(csv_string_to_list)
df_train['target'] = df_train['target'].apply(csv_string_to_list)

X_train = np.array(df_train['keypoints'].tolist(), dtype=np.float32)
y_train = np.array(df_train['target'].tolist(), dtype=np.float32)

# Reading Train CSV data
df_test = pd.read_csv('test_data.csv')
df_test['keypoints'] = df_test['keypoints'].apply(csv_string_to_list)
df_test['target'] = df_test['target'].apply(csv_string_to_list)

X_test = np.array(df_test['keypoints'].tolist(), dtype=np.float32)
y_test = np.array(df_test['target'].tolist(), dtype=np.float32)

# DataLoaders (train & test)
train_dataset = KeypointsDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# DataLoaders (train & test)
test_dataset = KeypointsDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

# Initialize model parameters
input_size = len(X_train[0])  # Features
output_size = 2  # Target (left_ankle, right_ankle)
loss_function = nn.MSELoss()
epochs = 10

# Hyperparameters testing
learning_rates = [0.1, 0.01, 0.001, 0.0001]
batch_sizes = [16, 32, 64, 128]
layer_configurations = [[64, 32], [128, 64, 32], [256, 128, 64, 32]]
activation_functions = [nn.ReLU, nn.Sigmoid, nn.Tanh]  # Exemples de fonctions d'activation

for lr in learning_rates:
    for batch_size in batch_sizes:
        for layers in layer_configurations:
            for activation_fn in activation_functions:
                print(f"Training with LR: {lr}, Batch Size: {batch_size}, Layers: {layers}, Activation: {activation_fn.__name__}")
                model = MLP(input_size, output_size, layers, activation_fn)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                # Training Loop
                for epoch in range(epochs):
                    model.train()
                    for inputs, targets in train_loader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = loss_function(outputs, targets)
                        loss.backward()
                        optimizer.step()

                # Model Evaluation
                model.eval()
                y_pred = []
                y_true = []
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        outputs = model(inputs)
                        y_pred.extend(outputs.tolist())
                        y_true.extend(targets.tolist())

                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                record_hyperparameters_performance(lr, batch_size, rmse, layers, activation_fn.__name__, epochs)