import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from model import PolicyNet, save_model, load_model


class CombinedDataset(Dataset):
    """Dataset combining human demos and MC demos."""

    def __init__(self, states, actions, weights=None):
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)
        self.weights = torch.FloatTensor(weights) if weights is not None else None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        if self.weights is not None:
            return self.states[idx], self.actions[idx], self.weights[idx]
        return self.states[idx], self.actions[idx]


def train_mc(
    human_data_path='data/human_demos.npz',
    mc_data_path='data/mc_demos.npz',
    init_policy_path='data/policy_imitation.pt',
    save_path='data/policy_mc.pt',
    mc_weight=2.0,
    epochs=100,
    batch_size=32,
    lr=5e-4,
    val_split=0.2,
    include_human_data=True,
    human_replay_fraction=0.0,
    early_stop_patience=15,
    lr_decay=0.5,
    min_lr=1e-5
):
    """
    Train policy on combined human + MC data.

    Args:
        human_data_path: path to human demonstrations
        mc_data_path: path to MC demonstrations
        init_policy_path: path to initial policy (optional, will train from scratch if not found)
        save_path: path to save improved policy
        mc_weight: weight multiplier for MC data (higher = more MC influence)
        epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
        val_split: fraction of data for validation
        include_human_data: whether to mix in human demonstrations
        human_replay_fraction: fraction of MC dataset size to sample from human demos even when include_human_data is False
        early_stop_patience: epochs without validation improvement before triggering LR decay or early stop
        lr_decay: multiplier applied to learning rate when patience is exceeded (<=1)
        min_lr: minimum allowable learning rate before stopping
    """
    print(f"\n=== Monte Carlo Policy Training ===")

    # Load human data
    human_states = np.array([])
    human_actions = np.array([])

    if include_human_data or human_replay_fraction > 0:
        if os.path.exists(human_data_path):
            human_data = np.load(human_data_path)
            human_states = human_data['states']
            human_actions = human_data['actions']
            print(f"Loaded {len(human_states)} human transitions")
        else:
            print("No human data found")
    if not include_human_data and human_replay_fraction <= 0:
        human_states = np.array([])
        human_actions = np.array([])
        print("Skipping human data (MC-focused training)")

    # Load MC data
    if not os.path.exists(mc_data_path):
        raise FileNotFoundError(f"MC data not found: {mc_data_path}")

    mc_data = np.load(mc_data_path)
    mc_states = mc_data['states']
    mc_actions = mc_data['actions']
    print(f"Loaded {len(mc_states)} MC transitions")

    # Combine datasets with weighting
    replay_states = np.array([])
    replay_actions = np.array([])

    if not include_human_data and human_replay_fraction > 0 and len(human_states) > 0:
        n_replay = max(1, int(len(mc_states) * human_replay_fraction))
        indices = np.random.choice(len(human_states), size=n_replay, replace=len(human_states) < n_replay)
        replay_states = human_states[indices]
        replay_actions = human_actions[indices]
        print(f"Injecting {n_replay} human transitions (~{human_replay_fraction*100:.1f}% of MC size)")

    if len(human_states) > 0 and include_human_data:
        # Repeat MC samples according to weight
        mc_repeat = max(1, int(round(mc_weight)))
        mc_states_weighted = np.repeat(mc_states, mc_repeat, axis=0)
        mc_actions_weighted = np.repeat(mc_actions, mc_repeat, axis=0)

        all_states = np.concatenate([human_states, mc_states_weighted])
        all_actions = np.concatenate([human_actions, mc_actions_weighted])

        print(f"Combined dataset: {len(all_states)} transitions "
              f"(human: {len(human_states)}, MC weighted: {len(mc_states_weighted)})")
    elif len(replay_states) > 0:
        all_states = np.concatenate([replay_states, mc_states], axis=0)
        all_actions = np.concatenate([replay_actions, mc_actions], axis=0)
        print(f"Combined dataset: {len(all_states)} transitions "
              f"(replay human: {len(replay_states)}, MC: {len(mc_states)})")
    else:
        all_states = mc_states
        all_actions = mc_actions
        print(f"Using only MC data: {len(all_states)} transitions")

    # Split train/val
    n_val = int(len(all_states) * val_split)
    indices = np.random.permutation(len(all_states))
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    train_states = all_states[train_indices]
    train_actions = all_actions[train_indices]
    val_states = all_states[val_indices]
    val_actions = all_actions[val_indices]

    # Create datasets
    train_dataset = CombinedDataset(train_states, train_actions)
    val_dataset = CombinedDataset(val_states, val_actions)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    state_dim = all_states.shape[1]
    n_actions = len(np.unique(all_actions))

    if os.path.exists(init_policy_path):
        print(f"Loading initial policy from {init_policy_path}")
        model = load_model(init_policy_path)
    else:
        print("Training from scratch")
        model = PolicyNet(state_dim=state_dim, n_actions=n_actions)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining on {len(train_states)} samples, validating on {len(val_states)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    best_val_acc = 0.0
    epochs_no_improve = 0

    # Training loop
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            states_batch, actions_batch = batch[0], batch[1]

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
            for batch in val_loader:
                states_batch, actions_batch = batch[0], batch[1]

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
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_model(model, save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                current_lr = optimizer.param_groups[0]['lr']
                new_lr = current_lr * lr_decay
                if new_lr >= min_lr and lr_decay < 1.0:
                    for group in optimizer.param_groups:
                        group['lr'] = new_lr
                    epochs_no_improve = 0
                    print(f"Patience exceeded, reducing LR to {new_lr:.6f}")
                else:
                    print("Early stopping due to validation plateau.")
                    break

    print(f"\n✓ Training complete!")
    print(f"✓ Best validation accuracy: {best_val_acc:.3f}")
    print(f"✓ Model saved to {save_path}")

    return model


if __name__ == '__main__':
    model = train_mc(
        human_data_path='data/human_demos.npz',
        mc_data_path='data/mc_demos.npz',
        init_policy_path='data/policy_imitation.pt',
        save_path='data/policy_mc.pt',
        mc_weight=2.0,
        epochs=100,
        batch_size=32,
        lr=5e-4
    )
