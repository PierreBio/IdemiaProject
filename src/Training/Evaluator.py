import torch
from sklearn.metrics import mean_squared_error
from math import sqrt


def evaluate_model(model, dataloader, device, loss_function):
    model.eval()
    losses = []
    y_pred, y_true = [], []

    with torch.no_grad():
        for _, _, inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            losses.append(loss.item())

            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(targets.cpu().numpy())

    root_mse = sqrt(mean_squared_error(y_true, y_pred))
    print(f"Evaluation RMSE: {root_mse:.4f}")
    return root_mse, losses
