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
    df['bbox'] = df['bbox'].apply(csv_string_to_list)
    df['keypoints'] = df['keypoints'].apply(csv_string_to_list)
    df['target'] = df['target'].apply(csv_string_to_list)
    return KeypointsDataset(df)


# -----------------------------------------------------------------------------
# MODEL LOGIC
# -----------------------------------------------------------------------------
# Setup
config = load_config(os.path.join(os.getcwd(), "config", "config.yaml"))
device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
print("Running on device: ", device)

models_path = os.path.join(os.getcwd(), "models")
if not os.path.exists(models_path):
    os.makedirs(models_path)

# Creating sub-folder for current run
exp_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
exp_learning_rate = config['training']['learning_rate']
exp_batch_size = config['training']['batch_size']
exp_name = f"{exp_timestamp}_LR{exp_learning_rate}_BS{exp_batch_size}"

exp_path = os.path.join("models", exp_name)
os.makedirs(exp_path, exist_ok=True)

# Train occlusion settings
occlusion_params = {
    'box_occlusion': {'occlusion_chance': 0.8, 'box_scale_factor': (0.5, 1)},
    'keypoints_occlusion': {'weight_position': "upper_body", 'weight_value': 0.7, 'min_visible_threshold': 5}
}

# Data Preparation
train_dataset = prepare_data(config['data']['train_path'])
val_dataset = prepare_data(config['data']['validation_path'])

train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

# Model Initialization
input_size = len(train_dataset[0][2])  # Features
model = MLP(input_size, config['model']['output_size'], config['model']['layers']).to(device)

# Training and retrieving best RMSE
optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
loss_function = torch.nn.MSELoss()
best_rmse, best_epoch = train_model(model, train_loader, val_loader, optimizer, loss_function, config['training']['epochs'], device, exp_path, occlusion_params)

# -----------------------------------------------------------------------------
# POST-TRAINING ACTIONS
# -----------------------------------------------------------------------------
# Saving Model
print("Saving model & model performances...")
model_filename = f"final_model_epoch_{best_epoch}_rmse_{best_rmse:.4f}.pth"
torch.save(model.state_dict(), os.path.join(exp_path, model_filename))
print(f"Model {model_filename} saved.")

# Saving Model performances
print("Adding Model performances in log file...")
performance_data = {
    "Timestamp": exp_timestamp,
    "Learning_Rate": exp_learning_rate,
    "Batch_Size": exp_batch_size,
    "Model_Layers": config['model']['layers'],
    "Configured Epochs": config['training']['epochs'],
    "Best Epoch": best_epoch,
    "RMSE": best_rmse
}
log_model_results(performance_data, csv_file=os.path.join(models_path, "model_performance_logs.csv"))
print("Model Performance saved.")
