import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import ast  # For safely evaluating strings as lists


def convert_string_to_list(string):
    try:
        return ast.literal_eval(string)
    except ValueError:
        return []  # or some default value


# Read data from CSV
df = pd.read_csv('occluded_only_w_threshold.csv')
df['keypoints'] = df['keypoints'].apply(convert_string_to_list)
df['target'] = df['target'].apply(convert_string_to_list)

X = np.array(df['keypoints'].tolist(), dtype=np.float32)
y = np.array(df['target'].tolist(), dtype=np.float32)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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


# Initialize model, loss function, and optimizer
input_size = len(X_train[0])  # Number of features in each input item
output_size = 2  # Target is a list of length 2
model = MLP(input_size, output_size)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create DataLoaders
train_dataset = KeypointsDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

test_dataset = KeypointsDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

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

# Evaluation
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

import matplotlib.pyplot as plt

# Assuming `y_test` are your actual values and `y_pred` are your model predictions
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs. Actual Values")
plt.show()
