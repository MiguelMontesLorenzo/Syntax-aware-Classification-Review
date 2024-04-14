import torch
from torch.utils.data import DataLoader
from src.utils import accuracy
import numpy as np
from typing import List


# from src.RecursiveModel.utils import Hook
from torch.utils.tensorboard import SummaryWriter


def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
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
    losses: List[float] = []
    accuracies: List[float] = []
    
    model.train()

    for sentences, labels, text_len in train_data:
        sentences: torch.Tensor = sentences.to(device)
        labels: torch.Tensor = labels.to(device)

        outputs: torch.Tensor = model(sentences, text_len)

        print("Labels sahpe", labels.shape)
        print("outputs", outputs.shape)
        print()
        loss_value: torch.nn.Module = loss(outputs, labels.long())
        
        accuracy_measure: torch.Tensor = accuracy(outputs, labels)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        losses.append(loss_value.item())
        accuracies.append(accuracy_measure.item())
        
    # write on tensorboard
    writer.add_scalar("train/loss", np.mean(losses), epoch)
    writer.add_scalar("train/accuracy", np.mean(accuracies), epoch)


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
        model: pytorch model.
        val_data: dataloader of validation data.
        loss: loss function.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """
    
    model.eval()

    with torch.no_grad():
        losses: List[float] = []
        accuracies: List[float] = []

        for sentences, labels, text_len in val_data:
            sentences: torch.Tensor = sentences.to(device)
            labels: torch.Tensor = labels.to(device)

            outputs: torch.Tensor = model(sentences, text_len)
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
            model: pytorch model.
            test_data: dataloader of test data.
            device: device of model.
            
        Returns:
            average accuracy.
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