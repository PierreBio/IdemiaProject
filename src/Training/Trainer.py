import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import plotly.graph_objects as go
import os
import random

from src.Training.Evaluator import evaluate_model
from src.ImageParser.ImageProcessor import ImageProcessor


def train_model(model, train_loader, val_loader, optimizer, loss_function: torch.nn, epochs, device, model_path, occlusion_params):
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
    val_loss_list = []

    # Initializing learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=5, verbose=True)

    for epoch in range(epochs):
        total_loss = 0

        for _, bbox, inputs, targets in train_loader:
            inputs, targets, bbox = inputs.to(device), targets.to(device), bbox.to(device)

            # Apply dynamic occlusion based on a random choice
            occlusion_type = random.choice(['no_occlusion', 'box_occlusion', 'keypoints_occlusion'])
            if occlusion_type == 'box_occlusion':
                bbox, inputs, targets = ImageProcessor.apply_box_occlusion_tensor(inputs,
                                                                                  bbox,
                                                                                  targets,
                                                                                  **occlusion_params.get("box_occlusion",
                                                                                                         {}))
            elif occlusion_type == 'keypoints_occlusion':
                inputs = ImageProcessor.apply_keypoints_occlusion_tensor(inputs,
                                                                         **occlusion_params.get("keypoints_occlusion",
                                                                                                {}))
            elif occlusion_type == 'no_occlusion':
                # do nothing
                pass

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
        val_rmse, eval_loss = evaluate_model(model, val_loader, device, loss_function)
        avg_val_loss = sum(eval_loss) / len(eval_loss)
        val_loss_list.append(avg_val_loss)

        # Update the learning rate based on validation RMSE
        scheduler.step(val_rmse)

        # TODO: add early stopping condition
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch + 1
            best_model_path = os.path.join(model_path, f"best_model_epoch_{epoch+1}_rmse_{best_rmse:.4f}.pth")
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch + 1}/{epochs} - Average Training Loss: {avg_loss:.4f}, Validation RMSE: {val_rmse:.4f}")

    # Save the training loss graph
    save_loss_graph_go(loss_list, model_path, "training")
    save_loss_graph_go(loss_list, model_path, "validation")
    return best_rmse, best_epoch


def save_loss_graph_go(loss_list, model_path, name):
    epochs = list(range(1, len(loss_list) + 1))

    # Graph Theme
    template_theme = "seaborn"

    # Figure settings
    trace_loss = go.Scatter(
        x=epochs, y=loss_list,
        mode='lines+markers',
        name=f'{name} Loss',
        marker=dict(color='MediumPurple'),
        line=dict(color='RebeccaPurple')
    )

    layout = go.Layout(
        title=f'{name} Performance',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Value', autorange=True),
        hovermode='closest',
        template=template_theme,  # Apply the chosen theme
        legend=dict(title='Metrics', x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)'),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig = go.Figure(data=[trace_loss], layout=layout)

    # Saving to HTML
    fig.write_html(os.path.join(model_path, f"{name}_performance.html"))
