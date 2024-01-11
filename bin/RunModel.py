import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import ast
import matplotlib.pyplot as plt


def csv_string_to_list(string):
    try:
        return ast.literal_eval(string)
    except ValueError:
        raise Exception("Could not process data, verify csv file")


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


# Reading CSV data
df = pd.read_csv('original_data.csv')
df['keypoints'] = df['keypoints'].apply(csv_string_to_list)
df['target'] = df['target'].apply(csv_string_to_list)

X_train = np.array(df['keypoints'].tolist(), dtype=np.float32)
y_train = np.array(df['target'].tolist(), dtype=np.float32)

# TODO: Train csv has original data only
# TODO: validation has occluded data -> same validation for every test
# TODO: Normalisation inputs (gaussian)
# Initialize model parameters
input_size = len(X_train[0])  # Features
output_size = 2  # Target (left_ankle, right_ankle)
model = MLP(input_size, output_size)

# Init Loss function
# TODO: use RMSE
loss_function = nn.MSELoss()

# Init Optimizer
# TODO: try different combination of hyperparams
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# DataLoaders (train & test)
train_dataset = KeypointsDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# Training Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

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
r2 = r2_score(y_true, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Plot test
plt.scatter(y_test, y_pred, color="blue", alpha=0.5, label='Predictions')
# plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs. Actual Values")
plt.show()
