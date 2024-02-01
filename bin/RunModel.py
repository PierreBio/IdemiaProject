import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import yaml
import datetime

from src.DataLoader.DataLoader import KeypointsDataset, csv_string_to_list
from src.Models.Mlp import MLP
from src.Training.Trainer import train_model
from src.Training.Evaluator import evaluate_model
from src.Common.utils import log_model_results


# -----------------------------------------------------------------------------
# load_config
# -----------------------------------------------------------------------------
def load_config(config_path):
    with open(config_path, 'r') as file:
        p_config = yaml.safe_load(file)
    return p_config


# -----------------------------------------------------------------------------
# prepare_data
# -----------------------------------------------------------------------------
def prepare_data(data_path):
    df = pd.read_csv(data_path)
    df['keypoints'] = df['keypoints'].apply(csv_string_to_list)
    df['target'] = df['target'].apply(csv_string_to_list)
    return KeypointsDataset(df)


# -----------------------------------------------------------------------------
# MODEL LOGIC
# -----------------------------------------------------------------------------
# Setup
config = load_config('../config/config.yaml')
device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
models_path = "../models/"
if not os.path.exists(models_path):
    os.makedirs(models_path)

# Creating sub-folder for current run
exp_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
exp_learning_rate = config['training']['learning_rate']
exp_batch_size = config['training']['batch_size']
exp_name = f"{exp_timestamp}_LR{exp_learning_rate}_BS{exp_batch_size}"

exp_path = os.path.join("../models/", exp_name)
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

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
train_model(model, train_loader, optimizer, loss_function, config['training']['epochs'], device, exp_path)

# Evaluation
root_mse = evaluate_model(model, test_loader, device)
print(f"Training completed. Final RMSE: {root_mse:.4f}")

# Saving model & other useful graphs/data
print("Saving model & model performances...")
model_filename = f"final_model_{root_mse:.4f}.pth"
torch.save(model.state_dict(), os.path.join(exp_path, model_filename))
print(f"Model {model_filename} saved.")

print("Adding Model performances in log file...")
performance_data = {
    'Timestamp': exp_timestamp,
    'Learning_Rate': exp_learning_rate,
    'Batch_Size': exp_batch_size,
    'Epochs': config['training']['epochs'],
    'Model_Layers': config['model']['layers'],
    'RMSE': root_mse
}
log_model_results(performance_data, csv_file=os.path.join(models_path, "model_performance_logs.csv"))
print("Model Performance saved.")
