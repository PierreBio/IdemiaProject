import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
import ast
import matplotlib.pyplot as plt
import random
import os
from src.ImageParser.ImagerProcessor import ImageProcessor
from src.Common.utils import apply_keypoints_occlusion, record_results

# Sets the device to GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set the path to save the model and loss curve
model_path = "models/"
if not os.path.exists(model_path):
    os.makedirs(model_path)


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
        self.bboxes = np.array([ast.literal_eval(bbox) for bbox in dataframe['bbox']], dtype=np.float32)

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        return self.img_ids[idx], self.keypoints[idx], self.targets[idx], self.bboxes[idx]


class MLP(nn.Module):
    def __init__(self, input_size, output_size, layers):
        """
        Initializes a Multi-Layer Perceptron with a specified number of layers,
        using ReLU as the activation function.

        Args:
            input_size (int): The size of the input features.
            output_size (int): The size of the output.
            layers (list of int): A list where each element is the size of a hidden layer.
        """
        super(MLP, self).__init__()
        self.m_layers = nn.Sequential()

        # Dynamically create hidden layers with ReLU activation
        for i, layer_size in enumerate(layers, start=1):
            self.m_layers.add_module(f"Linear_{i}", nn.Linear(input_size, layer_size))
            self.m_layers.add_module(f"Activation_{i}", nn.ReLU())

            # Update input_size for the next layer
            input_size = layer_size

        # Adding the output layer
        self.m_layers.add_module("Output", nn.Linear(input_size, output_size))

    def forward(self, x):
        return self.m_layers(x)


def train_and_evaluate(p_model, p_train_loader, p_test_loader, p_optimizer, p_epochs):
    loss_list = []

    p_model = p_model.to(device)
    for p_epoch in range(p_epochs):
        p_model.train()
        total_loss = 0
        for img_id, inputs, targets, bboxes in p_train_loader:
            occlusion_type = random.choice(["box", "keypoints"])

            # Apply chosen occlusion
            if occlusion_type == 'box':
                occluded_inputs = ImageProcessor.apply_box_occlusion(img_id, inputs, bboxes)
            else:  # keypoints
                occluded_inputs = apply_keypoints_occlusion(inputs, "upper_body")

            p_optimizer.zero_grad()
            occluded_inputs = occluded_inputs.to(device)
            targets = targets.to(device)
            outputs = p_model(occluded_inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            p_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(p_train_loader)
        loss_list.append(avg_loss)
        print(f"Epoch {p_epoch + 1}/{p_epochs} - Loss: {avg_loss:.4f}")

    p_model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for img_id, inputs, targets, bboxes in p_test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = p_model(inputs)
            y_pred.extend(outputs.tolist())
            y_true.extend(targets.tolist())

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"Evaluation completed - RMSE: {rmse:.4f}")
    return rmse, loss_list


def save_loss_graph(loss_list, lr, batch_size, layers):
    plt.plot(loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.savefig(model_path + f"loss_curve_{lr}_{batch_size}_{'-'.join(map(str, layers))}.png")
    plt.clf()


# Reading Train & Test CSV data
df_train = pd.read_csv('train_data.csv')
df_train['keypoints'] = df_train['keypoints'].apply(csv_string_to_list)
df_train['target'] = df_train['target'].apply(csv_string_to_list)

df_test = pd.read_csv('validation_data.csv')
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
loss_function = nn.MSELoss()

# Hyperparameters testing
epochs = 20
# learning_rates = [0.1, 0.01, 0.001, 0.0001]
learning_rates = [0.01]
# batch_sizes = [16, 32, 64, 128]
batch_sizes = [16]
# layer_configurations = [[64, 32], [128, 64, 32], [256, 128, 64, 32]]
layer_configurations = [[256, 128, 64, 32]]

# RMSE
best_rmse = float('inf')

# Grid search over hyperparameters
for lr in learning_rates:
    for batch_size in batch_sizes:
        for layers in layer_configurations:
            print(f"Training with epochs: {epochs}, learning rate: {lr}, batch size: {batch_size}, layers: {layers}")

            # Initialize model, dataloaders, and optimizer for each configuration
            model = MLP(input_size, output_size, layers)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Train and evaluate the model
            rmse, loss_l = train_and_evaluate(model, train_loader, test_loader, optimizer, epochs)

            # Save the graph for each configuration
            save_loss_graph(loss_l, lr, batch_size, layers)

            # Save the model with the best RMSE
            if rmse < best_rmse:
                torch.save(model.state_dict(), model_path + f"model_{lr}_{batch_size}_{'-'.join(map(str, layers))}.pt")
                best_rmse = rmse

            # Record the results
            performance_data = {
                "Learning Rate": lr,
                "Batch Size": batch_size,
                "Layers": "-".join(map(str, layers)),
                "RMSE": rmse,
                "Epochs": epochs
            }
            record_results(performance_data)
