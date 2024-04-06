import torch
from src.RecursiveModel.utils import get_batch, flatten
import numpy as np
from typing import List
from src.RecursiveModel.treebank import Tree
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
    root_only: bool = False,
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

    for i, batch in enumerate(get_batch(batch_size, train_data)):
        print(i)
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

    # Log training loss to TensorBoard
    writer.add_scalar("train/loss", np.mean(losses), epoch)


def val(
    model: torch.nn.Module,
    batch_size: int,
    val_data: List[Tree],
    device: str,
    loss_function: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    root_only=False,
) -> None:
    """
    Test step function.

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
    - None
    """
    model.eval()
    losses: list[float] = []

    with torch.no_grad():
        for _, batch in enumerate(get_batch(batch_size, val_data)):
            if root_only:
                labels = [tree.labels[-1] for tree in batch]
                labels = torch.autograd.Variable(
                    torch.tensor(labels, dtype=torch.long, device=device)
                )
            else:
                labels = [tree.labels for tree in batch]
                labels = torch.autograd.Variable(
                    torch.tensor(flatten(labels), dtype=torch.long, device=device)
                )

            # compute outputs and loss
            output = model(batch, root_only)
            loss_value = loss_function(output, labels.long())

            # add metrics to vectors
            losses.append(loss_value.item())

            # write on tensorboard
            writer.add_scalar("val/loss", np.mean(losses), epoch)


def test(model, test_data, root_only: bool = False):

    accuracy = 0
    num_node = 0

    for test in test_data:
        model.zero_grad()
        preds = model(test, root_only)
        labels = test.labels[-1:] if root_only else test.labels
        for pred, label in zip(preds.max(1)[1].data.tolist(), labels):
            num_node += 1
            if pred == label:
                accuracy += 1
    print("Test Acc: ", accuracy / num_node * 100)
    return accuracy / num_node * 100
