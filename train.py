import torch
import torch.nn as nn
from model import RPSCNN
from Data import get_dataloaders
import config

def train_model(train_dir, val_dir):
    # 1. Call get_dataloaders from data.py to load your datasets
    train_loader, val_loader = get_dataloaders(train_dir, val_dir, batch_size=config.BATCH_SIZE)

    # 2. Instantiate model, loss, optimizer, and scheduler
    model = RPSCNN(num_classes=len(config.CLASSES)).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    patience = 0
    best_val_loss = float("inf")

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"No Improvement: {patience}/{config.PATIENCE}")
        
        if patience == config.PATIENCE:
            print("Early Stopping triggered.")
            break
            
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # --- Training Loop ---
        model.train()
        correct_train, total_train = 0, 0
        train_batch_losses = []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            predicted = torch.argmax(output, dim=1)
            correct_train += (predicted == y_batch).sum().item()
            total_train += len(y_batch)
            train_batch_losses.append(loss.item())

        epoch_train_loss = sum(train_batch_losses) / len(train_batch_losses)
        epoch_train_acc = correct_train / total_train
        train_losses.append(round(epoch_train_loss, 2))
        train_accuracies.append(round(epoch_train_acc, 2))

        print(f"Train Loss: {epoch_train_loss:.2f}")
        print(f"Train Accuracy: {epoch_train_acc:.2f}")

        # --- Validation Loop ---
        model.eval()
        correct_val, total_val = 0, 0
        val_batch_losses = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
                output = model(X_batch)
                loss = criterion(output, y_batch)

                predicted = torch.argmax(output, dim=1)
                correct_val += (predicted == y_batch).sum().item()
                total_val += len(y_batch)
                val_batch_losses.append(loss.item())

        epoch_val_loss = sum(val_batch_losses) / len(val_batch_losses)
        epoch_val_acc = correct_val / total_val
        val_losses.append(round(epoch_val_loss, 2))
        val_accuracies.append(round(epoch_val_acc, 2))

        if epoch_val_loss < best_val_loss:
            patience = 0
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), config.MODEL_PATH)
            print("Saved as best model.")
        else:
            patience += 1

        print(f"Validation Loss: {epoch_val_loss:.2f}")
        print(f"Validation Accuracy: {epoch_val_acc:.2f}")

        scheduler.step()

    print("\nTrain Accuracies:", train_accuracies)
    print("Validation Accuracies:", val_accuracies)
    print("Train Losses:", train_losses)
    print("Validation Losses:", val_losses)

if __name__ == "__main__":
    train_model(train_dir="data/train", val_dir="data/validation")




# Output:
# Epoch 1/10
# No Improvement: 0/1
# Learning rate: 0.0007
# Train Loss: 0.44
# Train Accuracy: 0.82
# Saved as best model.
# Validation Loss: 0.61
# Validation Accuracy: 0.77

# Epoch 2/10
# No Improvement: 0/1
# Learning rate: 0.0007
# Train Loss: 0.02
# Train Accuracy: 1.00
# Validation Loss: 0.65
# Validation Accuracy: 0.87

# Epoch 3/10
# No Improvement: 1/1
# Early Stopping triggered.

# Train Accuracies: [0.82, 1.0]
# Validation Accuracies: [0.77, 0.87]
# Train Losses: [0.44, 0.02]
# Validation Losses: [0.61, 0.65]
