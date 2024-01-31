import matplotlib.pyplot as plt
import os


def train_model(model, train_loader, optimizer, loss_function, epochs, device, model_path):
    model.train()
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

        avg_loss = total_loss / len(train_loader)
        loss_list.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")

    save_loss_graph(loss_list, epochs, model_path)


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
