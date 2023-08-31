from timeit import default_timer as timer
from typing import Dict, Union

import torch
import torch.utils.data
import torch.utils.tensorboard
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassFBetaScore,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm import tqdm


def _train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Trains a PyTorch model for one epoch.

    Args:
        model (torch.nn.Module): Model to be trained.
        dataloader (torch.utils.data.DataLoader): Train DataLoader.
        loss_fn (torch.nn.Module): Loss function to be used.
        optimizer (torch.optim.Optimizer): Optimizer to be used.
        device (torch.device): Device to be used ("cpu", "cuda", "mps").

    Returns:
        Dict[str, float]: Dictionary containing the classification metrics (accuracy, precession, recall, f-beta score) and loss. Example:
            metrics = {
                "accuracy": 0.94,
                "precession": 0.80,
                "recall": 0.69,
                "fscore": 0.76,
                "loss": 0.18,
            }
    """
    model.train()

    train_loss = 0
    train_metrics = {
        "accuracy": MulticlassAccuracy(num_classes=2).to(device),
        "precession": MulticlassPrecision(num_classes=2).to(device),
        "recall": MulticlassRecall(num_classes=2).to(device),
        "fscore": MulticlassFBetaScore(num_classes=2, beta=2.0).to(device),
    }

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = y_pred.argmax(dim=1)

        for key, _ in train_metrics.items():
            train_metrics[key].update(y_pred_class, y)

    train_loss = train_loss / len(dataloader)

    for key, _ in train_metrics.items():
        train_metrics[key] = train_metrics[key].compute().item()
    train_metrics["loss"] = train_loss

    return train_metrics


def _test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Tests a PyTorch model for one epoch.

    Args:
        model (torch.nn.Module): Model to be tested.
        dataloader (torch.utils.data.DataLoader): Test DataLoader.
        loss_fn (torch.nn.Module): Loss function to be used.
        device (torch.device): Device to be used ("cpu", "cuda", "mps").

    Returns:
        Dict[str, float]: Dictionary containing the classification metrics (accuracy, precession, recall, f-beta score) and loss. Example:
            metrics = {
                "accuracy": 0.94,
                "precession": 0.80,
                "recall": 0.69,
                "fscore": 0.76,
                "loss": 0.18,
            }
    """
    model.eval()

    test_loss = 0
    test_metrics = {
        "accuracy": MulticlassAccuracy(num_classes=2).to(device),
        "precession": MulticlassPrecision(num_classes=2).to(device),
        "recall": MulticlassRecall(num_classes=2).to(device),
        "fscore": MulticlassFBetaScore(num_classes=2, beta=2.0).to(device),
    }

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            test_loss += loss_fn(test_pred_logits, y)

            test_pred_labels = test_pred_logits.argmax(dim=1)

            for key, _ in test_metrics.items():
                test_metrics[key].update(test_pred_labels, y)

    test_loss = test_loss / len(dataloader)

    for key, _ in test_metrics.items():
        test_metrics[key] = test_metrics[key].compute().item()
    test_metrics["loss"] = test_loss

    return test_metrics


def _display_metrics(
    phase: str,
    epoch: int,
    metrics: Dict[str, float],
    writer: Union[torch.utils.tensorboard.writer.SummaryWriter, None] = None,
) -> None:
    """Receives data about the phase and epoch and prints out the metrics associated with them. Optionally, it can also save the metrics" evolution throughout training/testing to a tensorboard writer.

    Args:
        phase (str): The current phase of "train" or "test".
        epoch (int): The current epoch on "train" phase or total number of epochs on "test" phase.
        metrics (Dict[str, float]): Dictionary containing the classification metrics and loss.
        writer (Union[torch.utils.tensorboard.writer.SummaryWriter, None], optional): Tensorboard
            SummaryWriter object. If it is None then metrics will not be saved to tensorboard
            Defaults to None.

    Raises:
        ValueError: If the phase is not one of "train" or "test".
    """
    supported_phases = ["train", "test"]
    if phase not in supported_phases:
        raise ValueError(f"Phase {phase} not supported. Please choose between {supported_phases}")

    # Print Metrics
    print(f"{phase} || epoch: {epoch} | ", end="")

    for key, metric in metrics.items():
        print(f"{key}: {metric:.4f} | ", end="")
    print()

    # Save Metrics to Tensorboard
    if writer is not None:
        for key, metric in metrics.items():
            writer.add_scalars(
                main_tag=f"{phase}",
                tag_scalar_dict={
                    f"{key}": metric,
                },
                global_step=epoch
            )
        writer.close()


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    writer: Union[torch.utils.tensorboard.writer.SummaryWriter, None] = None
) -> Dict[str, float]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through _train_step() function for a number of epochs. 
    When the training is done, passes the same updated model through _test_step() function,
    to evaluate the model's performance on the test dataset.

    Calculates, prints and stores evaluation metrics throughout.

    Stores metrics to specified writer in debug/runs/ folder.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on ("cuda", "cpu", "cuda").
      writer: A SummaryWriter() instance to log model results to.

    Returns:
      Dict[str, float]: A dictionary containing the classification metrics, the loss for the test 
      dataset, and the time taken to train it. Example:
            metrics = {
                "accuracy": 0.94,
                "precession": 0.80,
                "recall": 0.69,
                "fscore": 0.76,
                "loss": 0.18,
                "time": 0.18,
            }
    """
    start_time = timer()

    for epoch in tqdm(range(epochs)):
        train_metrics = _train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        _display_metrics(
            phase="train",
            epoch=epoch,
            metrics=train_metrics,
            writer=writer
        )

    test_metrics = _test_step(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device
    )

    _display_metrics(
        phase="test",
        epoch=epochs,
        metrics=test_metrics,
        writer=writer
    )

    end_time = timer()

    metrics = test_metrics
    metrics["time"] = end_time - start_time

    return metrics
