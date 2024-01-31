import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
from src.DataLoader.DataLoader import KeypointsDataset, csv_string_to_list
from src.Models.Mlp import MLP
from src.Training.Trainer import train_model
from src.Training.Evaluator import evaluate_model
import yaml


# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        p_config = yaml.safe_load(file)
    return p_config


def prepare_data(data_path):
    df = pd.read_csv(data_path)
    df['keypoints'] = df['keypoints'].apply(csv_string_to_list)
    df['target'] = df['target'].apply(csv_string_to_list)
    return KeypointsDataset(df)


# Setup
config = load_config('../config/config.yaml')
device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
model_path = "../models/"
if not os.path.exists(model_path):
    os.makedirs(model_path)

# Data Preparation
train_dataset = prepare_data(config['data']['train_path'])
test_dataset = prepare_data(config['data']['validation_path'])

train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

# Model Initialization
input_size = len(train_dataset[0][1])  # Features
model = MLP(input_size, config['model']['output_size'], config['model']['layers']).to(device)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
loss_function = torch.nn.MSELoss()

train_model(model, train_loader, optimizer, loss_function, config['training']['epochs'], device, model_path)

# Evaluation
root_mse = evaluate_model(model, test_loader, device)

# Optionally save the model
torch.save(model.state_dict(), os.path.join(model_path, "final_model.pth"))

print(f"Training completed. Final RMSE: {root_mse:.4f}")
