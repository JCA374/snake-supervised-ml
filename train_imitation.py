import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from model import PolicyNet, save_model


class SnakeDataset(Dataset):
    """Dataset for Snake (state, action) pairs."""

    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


def train_imitation(
    data_path='data/human_demos.npz',
    save_path='data/policy_imitation.pt',
    epochs=100,
    batch_size=32,
    lr=1e-3,
    val_split=0.2
):
    """
    Train policy network via imitation learning.

    Args:
        data_path: path to recorded demonstrations
        save_path: path to save trained model
        epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
        val_split: fraction of data for validation
    """
    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = np.load(data_path)
    states = data['states']
    actions = data['actions']

    print(f"\n=== Imitation Learning ===")
    print(f"Loaded {len(states)} transitions from {data_path}")
    print(f"State shape: {states[0].shape}")
    print(f"Action distribution: {np.bincount(actions)}")

    # Split train/val
    n_val = int(len(states) * val_split)
    indices = np.random.permutation(len(states))
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    train_states = states[train_indices]
    train_actions = actions[train_indices]
    val_states = states[val_indices]
    val_actions = actions[val_indices]

    # Create datasets
    train_dataset = SnakeDataset(train_states, train_actions)
    val_dataset = SnakeDataset(val_states, val_actions)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    state_dim = states.shape[1]
    n_actions = len(np.unique(actions))
    model = PolicyNet(state_dim=state_dim, n_actions=n_actions)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining on {len(train_states)} samples, validating on {len(val_states)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    best_val_acc = 0.0

    # Training loop
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for states_batch, actions_batch in train_loader:
            optimizer.zero_grad()

            logits = model(states_batch)
            loss = criterion(logits, actions_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == actions_batch).sum().item()
            train_total += len(actions_batch)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for states_batch, actions_batch in val_loader:
                logits = model(states_batch)
                loss = criterion(logits, actions_batch)

                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == actions_batch).sum().item()
                val_total += len(actions_batch)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_model(model, save_path)

    print(f"\n✓ Training complete!")
    print(f"✓ Best validation accuracy: {best_val_acc:.3f}")
    print(f"✓ Model saved to {save_path}")

    return model


if __name__ == '__main__':
    model = train_imitation(
        data_path='data/human_demos.npz',
        save_path='data/policy_imitation.pt',
        epochs=100,
        batch_size=32,
        lr=1e-3
    )
