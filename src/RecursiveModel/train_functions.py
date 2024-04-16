import torch
from src.RecursiveModel.utils import get_batch, flatten, accuracy
import numpy as np
from typing import List
from src.RecursiveModel.treebank import Tree

# from src.RecursiveModel.utils import Hook
from torch.utils.tensorboard import SummaryWriter


def train(
    model: torch.nn.Module,
    batch_size: int,
    train_data: List[Tree],
    device: str,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    root_only: bool = False
) -> None:
    """
    Train step function.

    Args:
    - model (torch.nn.Module): model to train.
    - batch_size (int): batch_size.
    - train_data (List[Tree]): training tree.
    - device (str): model device.
    - optimizer (torch.optim.Optimizer): optimizer.
    - loss_function (torch.nn.Module): loss.
    - writer (SummaryWriter): Tensorboard writer.
    - epoch (int): epoch.
    - root_only(bool).

    Returns:
    - None
    """

    model.train()
    losses: list[float] = []
    accuracies: List[float] = []

    for _, batch in enumerate(get_batch(batch_size, train_data)):
        if root_only:
            labels_list: List[int] = [tree.labels[-1] for tree in batch]
            labels: torch.autograd.Variable = torch.autograd.Variable(
                torch.tensor(labels_list, dtype=torch.long, device=device)
            )

        else:
            labels_list: List[int] = [tree.labels for tree in batch]
            labels: torch.autograd.Variable = torch.autograd.Variable(
                torch.tensor(flatten(labels_list), dtype=torch.long, device=device)
            )

        output: torch.Tensor = model(batch, root_only)
        loss: float = loss_function(output, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        losses.append(loss.item())

        accuracy_value: float = accuracy(output, labels)

        accuracies.append(accuracy_value)

    acc: float = torch.mean(torch.tensor(accuracies))


    # Log training loss to TensorBoard
    writer.add_scalar("train/loss", np.mean(losses), epoch)
    writer.add_scalar("train/accuracy", acc, epoch)


def val(
    model: torch.nn.Module,
    batch_size: int,
    val_data: List[Tree],
    device: str,
    loss_function: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    root_only=False,
) -> float:
    """
    Val step function.

    Args:
    - model (torch.nn.Module): model to train.
    - batch_size (int): batch_size.
    - val_data (List[Tree]): training tree.
    - device (str): model device.
    - loss_function (torch.nn.Module): loss.
    - writer (SummaryWriter): Tensorboard writer.
    - epoch (int): epoch.
    - root_only(bool).

    Returns:
    - val_loss (float): loss in that iteration
    """
    model.eval()
    losses: list[float] = []
    accuracies: List[float] = []

    with torch.no_grad():
        for _, batch in enumerate(get_batch(batch_size, val_data)):
            if root_only:
                labels_list: List[int] = [tree.labels[-1] for tree in batch]
                labels: torch.autograd.Variable = torch.autograd.Variable(
                    torch.tensor(labels_list, dtype=torch.long, device=device)
                )

            else:
                labels_list: List[int] = [tree.labels for tree in batch]
                labels: torch.autograd.Variable = torch.autograd.Variable(
                    torch.tensor(flatten(labels_list), dtype=torch.long, device=device)
                )

            # compute outputs and loss
            output: torch.Tensor = model(batch, root_only)
            loss_value: float = loss_function(output, labels.long())

            # add metrics to vectors
            losses.append(loss_value.item())

            accuracy_value: float = accuracy(output, labels)

            accuracies.append(accuracy_value)

        acc: float = torch.mean(torch.tensor(accuracies))

        # write on tensorboard
        writer.add_scalar("val/loss", np.mean(losses), epoch)
        writer.add_scalar("val/accuracy", acc, epoch)

    return loss_value.item()


def test(
    model: torch.nn.Module,
    batch_size: int,
    test_data: List[Tree],
    device: str,
    root_only=False,
) -> float:
    """
    Test step function.

    Args:
    - model (torch.nn.Module): model to train.
    - batch_size (int): batch_size.
    - test_data (List[Tree]): training tree.
    - device (str): model device.
    - root_only(bool).

    Returns:
    - None
    """
    model.eval()
    accuracies: List[float] = []

    with torch.no_grad():
        for _, batch in enumerate(get_batch(batch_size, test_data)):
            if root_only:
                labels_list: List[int] = [tree.labels[-1] for tree in batch]
                labels: torch.autograd.Variable = torch.autograd.Variable(
                    torch.tensor(labels_list, dtype=torch.long, device=device)
                )
            else:
                labels_list: List[int] = [tree.labels for tree in batch]
                labels: torch.autograd.Variable = torch.autograd.Variable(
                    torch.tensor(flatten(labels_list), dtype=torch.long, device=device)
                )

            # compute outputs and loss
            outputs = model(batch, root_only).to(device)

            accuracy_value: float = accuracy(outputs, labels)

            accuracies.append(accuracy_value)

    acc: float = torch.mean(torch.tensor(accuracies))

    return acc
