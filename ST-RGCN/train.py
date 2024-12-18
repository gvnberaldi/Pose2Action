import os
import sys

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, ConcatDataset
import wandb
sys.path.append(os.path.dirname(__file__))
from data.dataset import SpatioTemporalGraphDataset
from model.model import RGCN


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_type, data.batch)

        # Compute loss
        loss = criterion(output, data.label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Compute accuracy
        predictions = output.argmax(dim=1)
        correct = (predictions == data.label).sum().item()
        total_correct += correct
        total_samples += data.label.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def test(model, device, test_loader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_type, data.batch)

            # Compute loss
            total_loss += criterion(output, data.label).item()

            # Compute accuracy
            predictions = output.argmax(dim=1)
            correct = (predictions == data.label).sum().item()
            total_correct += correct
            total_samples += data.label.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


if __name__ == "__main__":
    # Initialize wandb
    wandb.login(key='e5e7b3c0c3fbc088d165766e575853c01d6cb305')
    wandb.init(project="rgcn-action-classification", entity="gvnberaldi")

    # Log hyperparameters
    hyperparameters = {
        "in_channels": 4,
        "hidden_channels": 32,
        "out_channels": 64,
        "mlp_dim": 32,
        "batch_size": 4,
        "learning_rate": 0.001,
        "epochs": 100
    }
    wandb.config.update(hyperparameters)

    # Assuming you have a SpatioTemporalGraphDataset instance 'dataset' already defined
    dataset = SpatioTemporalGraphDataset(root=os.path.join(os.getcwd(), 'dataset\\BAD'),
                                         activity_label_file=os.path.join(os.getcwd(), 'dataset\\activity_labels.txt'))

    # Split dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Initialize DataLoader for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 8
    num_relations = 2
    # Define model
    model = RGCN(in_channels=hyperparameters['in_channels'], hidden_channels=hyperparameters['hidden_channels'],
                 out_channels=hyperparameters['out_channels'], mlp_dim=hyperparameters['mlp_dim'],
                 num_classes=num_classes, num_relations=num_relations).to(device)
    # Print the model architecture
    print(model)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(0, hyperparameters['epochs']):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, criterion)
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Evaluate on test set
        test_loss, test_accuracy = test(model, device, test_loader, criterion)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Log metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

    # Finish the wandb run
    wandb.finish()
