# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader

# own modules
from src.utils import accuracy
from src.VecAvgModel.models import Weighter


def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    weigther: Weighter,
) -> None:
    """
    This function computes the training step.

    Args:
        model: pytorch model.
        train_data: train dataloader.
        loss: loss function.
        optimizer: optimizer object.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """

    # define metric lists
    losses: list[float] = []
    accuracies: list[float] = []

    # Set the model in training mode
    model.train()

    # Iterate over the training data
    for X_train_batch, y_train_batch, _ in train_data:
        # Move the data to the correct device
        X_train: torch.Tensor
        y_train: torch.Tensor
        X_train, y_train = X_train_batch.to(device), y_train_batch.to(device)

        # Flatten the input tensor
        X_train_flatten: torch.Tensor = X_train.view(X_train.size(0), -1)

        # 1. Produce predictions
        y_pred: torch.Tensor = model(X_train_flatten, weigther)

        # 2. Compute loss
        batch_loss: float = loss(y_pred, y_train)
        losses.append(batch_loss)

        # 3. Compute accuracy
        acc: torch.Tensor = accuracy(y_pred, y_train)
        accuracies.append(acc.item())

        # 4. Zero grad of the optimizer
        optimizer.zero_grad()

        # 5. Loss backwards
        batch_loss.backward()

        # 6. Progress the optimizer
        optimizer.step()

    # display metrics
    print(f"\nTrain loss [epoch: {epoch + 1}]: {np.mean(losses)}")
    print(f"Train accuracy [epoch: {epoch + 1}]: {np.mean(accuracies)}")


def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    epoch: int,
    device: torch.device,
    weighter: Weighter,
) -> None:
    """
    This function computes the validation step.

    Args:
        model: pytorch model.
        val_data: dataloader of validation data.
        loss: loss function.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """

    # define metric lists
    losses: list[float] = []
    accuracies: list[float] = []

    # Set the model in evaluation mode
    model.eval()

    # Iterate over the training data
    with torch.no_grad():
        for X_val_batch, y_val_batch, _ in val_data:
            # Move the data to the correct device
            X_val: torch.Tensor
            y_val: torch.Tensor
            X_val, y_val = X_val_batch.to(device), y_val_batch.to(device)

            # Flatten the input tensor
            X_val_flatten: torch.Tensor = X_val.view(X_val.size(0), -1)

            # 1. Produce predictions
            y_pred: torch.Tensor = model(X_val_flatten, weighter)

            # 2. Compute loss
            batch_loss: float = loss(y_pred, y_val)
            losses.append(batch_loss)

            # 3. Compute accuracy
            acc: torch.Tensor = accuracy(y_pred, y_val)
            accuracies.append(acc.item())

    # display metrics
    print(f"Validation loss [epoch: {epoch + 1}]: {np.mean(losses)}")
    print(f"Validation accuracy [epoch: {epoch + 1}]: {np.mean(accuracies)}")


def test_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
    weighter: Weighter,
) -> float:
    """
    This function computes the test step.

    Args:
        model: pytorch model.
        val_data: dataloader of test data.
        device: device of model.

    Returns:
        average accuracy.
    """

    # define metric lists
    accuracies: list[float] = []

    # Set the model in evaluation mode
    model.eval()

    # Iterate over the training data
    for X_test_batch, y_test_batch in test_data:
        # Move the data to the correct device
        X_test: torch.Tensor
        y_test: torch.Tensor
        X_test, y_test = X_test_batch.to(device), y_test_batch.to(device)

        # Flatten the input tensor
        X_test_flatten: torch.Tensor = X_test.view(X_test.size(0), -1)

        # 1. Produce predictions
        y_pred: torch.Tensor = model(X_test_flatten, weighter)

        # 2. Compute accuracy
        acc: torch.Tensor = accuracy(y_pred, y_test)
        accuracies.append(acc.item())

    return np.mean(accuracies)
