import matplotlib.pyplot as plt
import os
import torch

from src.Training.Evaluator import evaluate_model


def train_model(model, train_loader, val_loader, optimizer, loss_function: torch.nn, epochs, device, model_path):
    """
    Train the model, evaluate it on the validation set after each epoch,
    and keep track of the best Root MSE.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        loss_function (torch.nn): Loss function used in training.
        epochs (int): Number of epochs to train.
        device (torch.device): Device on which to train the model.
        model_path (str): Path where the best model will be saved.

    Returns:
        float: The best Root MSE achieved on the validation set during training.
    """
    model.train()
    best_rmse = float('inf')
    best_epoch = -1
    loss_list = []

    for epoch in range(epochs):
        total_loss = 0

        for _, inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Adding Average Loss
        avg_loss = total_loss / len(train_loader)
        loss_list.append(avg_loss)

        # Evaluate on validation set
        val_rmse = evaluate_model(model, val_loader, device)
        # TODO: add early stopping condition?
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch + 1
            best_model_path = os.path.join(model_path, f"best_model_epoch_{epoch+1}_rmse_{best_rmse:.4f}.pth")
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch + 1}/{epochs} - Average Training Loss: {avg_loss:.4f}, Validation RMSE: {val_rmse:.4f}")

    # Save the training loss graph
    save_loss_graph(loss_list, epochs, model_path)

    return best_rmse, best_epoch


def save_loss_graph(loss_list, epochs, model_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), loss_list, marker='o', label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_path, "training_loss_curve.png"))
    plt.close()
