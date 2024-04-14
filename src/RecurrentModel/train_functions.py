import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import List
try:
    from src.RecurrentModel.utils import accuracy
except ImportError:
    from utils import accuracy


def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> float:
    """
    This function computes the training step.

    Args:
    - model (torch.nn.Module): pytorch model.
    - train_data (DataLoader): train dataloader.
    - loss (torch.nn.Module): loss function.
    - optimizer (torch.optim.Optimizer): optimizer object.
    - writer (SummaryWriter): tensorboard writer.
    - epoch (int): epoch number.
    - device (torch.device): device of model.

    Return:
    - final_loss (float): average loss for the early stopping.
    """

    # Define metric lists
    losses: List[float] = []
    accuracies: List[float] = []
    
    model.train()

    for sentences, labels, text_len in train_data:
        sentences: torch.Tensor = sentences.to(device)
        labels: torch.Tensor = labels.to(device)

        outputs: torch.Tensor = model(sentences, text_len)

        # print("Labels sahpe", labels.shape)
        # print("outputs", outputs.shape)
        # print()
        loss_value: torch.nn.Module = loss(outputs, labels.long())
        
        accuracy_measure: torch.Tensor = accuracy(outputs, labels)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        losses.append(loss_value.item())
        accuracies.append(accuracy_measure.item())
    
    final_loss: float = float(np.mean(losses))

    # write on tensorboard
    writer.add_scalar("train/loss", final_loss, epoch)
    writer.add_scalar("train/accuracy", np.mean(accuracies), epoch)

    return final_loss


def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function computes the validation step.

    Args:
    - model (torch.nn.Module): pytorch model.
    - val_data (DataLoader): val dataloader.
    - loss (torch.nn.Module): loss function.
    - writer (SummaryWriter): tensorboard writer.
    - epoch (int): epoch number.
    - device (torch.device): device of model.

    Return:
    - None
    """
    
    model.eval()

    with torch.no_grad():
        losses: List[float] = []
        accuracies: List[float] = []

        for sentences, labels, text_len in val_data:
            sentences: torch.Tensor = sentences.to(device)
            labels: torch.Tensor = labels.to(device)

            outputs: torch.Tensor = model(sentences, text_len)

            # print("Labels sahpe", labels.shape)
            # print("outputs", outputs.shape)
            # print()
            loss_value: torch.nn.Module = loss(outputs, labels.long())
            
            accuracy_measure: torch.Tensor = accuracy(outputs, labels)

            losses.append(loss_value.item())
            accuracies.append(accuracy_measure.item())
        
        writer.add_scalar("val/loss", np.mean(losses), epoch)
        writer.add_scalar("val/accuracy", np.mean(accuracies), epoch)


def t_step(
        model: torch.nn.Module,
        test_data: DataLoader,
        device: torch.device,
    ) -> float:
        """
        This function computes the test step.

        Args:
        - model (torch.nn.Module): pytorch model.
        - test_data (DataLoader): test dataloader.
        - device (torch.device): device of model.
            
        Returns:
        - final_accuracy (float): average accuracy.
        """

        model.eval()
        accuracies: List[float] = []

        with torch.no_grad():
            
            for sentences, labels, text_len in test_data:
                sentences: torch.Tensor = sentences.to(device)
                labels: torch.Tensor = labels.to(device)

                outputs: torch.Tensor = model(sentences, text_len)
                
                accuracy_measure: torch.Tensor = accuracy(outputs, labels)

                accuracies.append(accuracy_measure.item())
            
            final_accuracy: float = float(np.mean(accuracies))
        return final_accuracy