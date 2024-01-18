import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import ast
import matplotlib.pyplot as plt
import random

from src.Common.utils import apply_box_occlusion, apply_keypoints_occlusion

def csv_string_to_list(string):
    try:
        return ast.literal_eval(string)
    except ValueError:
        raise Exception("Could not process data, verify csv file")


# Define the Dataset Class
class KeypointsDataset(Dataset):
    def __init__(self, dataframe):
        self.keypoints = np.array(dataframe['keypoints'].tolist(), dtype=np.float32)
        self.targets = np.array(dataframe['target'].tolist(), dtype=np.float32)
        self.img_ids = dataframe['img_id'].values

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        return self.img_ids[idx], self.keypoints[idx], self.targets[idx]


# Define the Neural Network Model
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.layers(x)


# Reading Train & Test CSV data
df_train = pd.read_csv('train_data_original.csv')
df_train['keypoints'] = df_train['keypoints'].apply(csv_string_to_list)
df_train['target'] = df_train['target'].apply(csv_string_to_list)

df_test = pd.read_csv('test_data.csv')
df_test['keypoints'] = df_test['keypoints'].apply(csv_string_to_list)
df_test['target'] = df_test['target'].apply(csv_string_to_list)

# DataLoaders (train & test)
train_dataset = KeypointsDataset(df_train)

# DataLoaders (train & test)
test_dataset = KeypointsDataset(df_test)

# DataLoaders
batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model parameters
input_size = len(train_dataset[0][1])  # Features
output_size = 2  # Target (left_ankle, right_ankle)
model = MLP(input_size, output_size)
loss_function = nn.MSELoss()
epochs = 10

# Hyperparameters testing
learning_rates = [0.1, 0.01, 0.001, 0.0001]
batch_sizes = [16, 32, 64, 128]

# Best Hyperparameters init
best_rmse = float('inf')
best_lr = None
best_batch_size = None

# Grid search over learning rates and batch sizes
for lr in learning_rates:
    for batch_size in batch_sizes:
        print(f"Training with learning rate: {lr}, batch size: {batch_size}")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training Loop
        for epoch in range(epochs):
            model.train()
            for img_id, inputs, targets in train_loader:
                # Randomly choose an occlusion method
                occlusion_type = random.choice(["box", "keypoints"])

                # Apply chosen occlusion
                if occlusion_type == 'box':
                    occluded_inputs = apply_box_occlusion(img_id, inputs)
                else:  # keypoints
                    occluded_inputs = apply_keypoints_occlusion(inputs, "upper_body")

                optimizer.zero_grad()
                outputs = model(occluded_inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

        # Model Evaluation
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for img_id, inputs, targets in test_loader:
                outputs = model(inputs)
                y_pred.extend(outputs.tolist())
                y_true.extend(targets.tolist())

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        print(f"RMSE: {rmse}")

        # Best hyperparameters update
        if rmse < best_rmse:
            best_rmse = rmse
            best_lr = lr
            best_batch_size = batch_size

# Output the best hyperparameter set
print(f"Best Hyperparameters => Learning rate: {best_lr}, Batch size: {best_batch_size}, RMSE: {best_rmse}")

# TODO: plot loss curve
# # Plot test
# plt.scatter(y_test, y_pred, color="blue", alpha=0.5, label='Predictions')
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.title("Predicted vs. Actual Values")
# plt.show()
