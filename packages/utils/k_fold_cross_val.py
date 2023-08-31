from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, Union

import numpy as np
import torch.utils.data
import torch.utils.tensorboard
import torchvision.datasets
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassFBetaScore,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm import tqdm

from packages.models.tiny_vgg import TinyVGG
from packages.utils.storage import save_model


def _cross_validation_train(
    model: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> Dict[str, float]:
    """Trains the model on the train_loader data using loss_fn as a loss function and optimizer to update model's weights.

    Args:
        model (torch.nn.Module): Model to be trained (must have 2 output units).
        device (torch.device): Device on which the calculations will be performed "cpu", "cuda" 
            or "mps".
        train_loader (torch.utils.data.DataLoader): DataLoader object containing the training data.
        loss_fn (torch.nn.Module): Loss function to be used.
        optimizer (torch.optim.Optimizer): Optimizer to be used.

    Returns:
        Dict[str, float]: Dictionary containing the classification metrics (accuracy, precession, recall, f-beta score) and loss. Example:
            metrics = {
                "accuracy": 0.92,
                "precession": 0.84,
                "recall": 0.61,
                "fscore": 0.78,
                "loss": 0.12,
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

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)

        loss = loss_fn(output, target)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_class = output.argmax(dim=1)

        for key, _ in train_metrics.items():
            train_metrics[key].update(pred_class, target)

    train_loss = train_loss / len(train_loader)

    for key, _ in train_metrics.items():
        train_metrics[key] = train_metrics[key].compute().item()
    train_metrics["loss"] = train_loss

    return train_metrics


def _cross_validation_test(
    model: torch.nn.Module,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
) -> Dict[str, float]:
    """Evaluate the model on the test_loader data suing the loss_fn as the loss function. It can be used on every set that does not result updating the model's parameters (e.g. validation and test data sets).

    Args:
        model (torch.nn.Module): Model to be evaluated (must have 2 output units).
        device (torch.device): Device on which the calculations will be performed "cpu", "cuda" 
            or "mps".
        train_loader (torch.utils.data.DataLoader): DataLoader object containing the data.
        loss_fn (torch.nn.Module): Loss function to be used.

    Returns:
        Dict[str, float]: Dictionary containing the classification metrics (accuracy, precession, recall, f-beta score) and loss. Example:
            metrics = {
                "accuracy": 0.95,
                "precession": 0.83,
                "recall": 0.68,
                "fscore": 0.74,
                "loss": 0.10,
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
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += loss_fn(output, target).item()

            pred = output.argmax(dim=1)

            for key, _ in test_metrics.items():
                test_metrics[key].update(pred, target)

    test_loss = test_loss / len(test_loader)

    for key, _ in test_metrics.items():
        test_metrics[key] = test_metrics[key].compute().item()
    test_metrics["loss"] = test_loss

    return test_metrics


def _get_model(
    model_name: str,
    hidden_units: int
) -> torch.nn.Module:
    """Returns a model based on the model_name and hidden_units parameters. The list of available models is: "tiny_vgg".

    Args:
        model_name (str): String that identifies the model to be used.
        hidden_units (int): Number of hidden units to be used in the model.

    Raises:
        ValueError: If the model_name is not one of "tiny_vgg".

    Returns:
        torch.nn.Module: The model to be used.
    """
    if model_name == "tiny_vgg":
        model = TinyVGG(
            input_shape=3,
            hidden_units=hidden_units,
            output_shape=2
        )
    else:
        raise ValueError(
            f"Model name {model_name} is not supported. "
            "Please choose between 'tiny_vgg'."
        )

    return model


def _get_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    learning_rate: float = 1e-3,
) -> torch.optim.Optimizer:
    """Returns an optimizer based on the optimizer_name and learning_rate parameters. The list of available optimizers is: "adam", "sgd".

    Args:
        model (torch.nn.Module): Model to be optimized (the parameters of it are needed).
        optimizer_name (str): String that identifies the optimizer to be used.
        learning_rate (float, optional): Learning rate to be used in updates. Defaults to 1e-3.

    Raises:
        ValueError: If the optimizer_name is not one of "adam", "sgd".

    Returns:
        torch.optim.Optimizer: The optimizer to be used.
    """
    if optimizer_name == "adam":
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=learning_rate
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=learning_rate
        )
    else:
        raise ValueError(
            f"Optimizer name {optimizer_name} is not supported. "
            "Please choose between 'adam' and 'sgd'."
        )

    return optimizer


def _display_metrics(
    phase: str,
    fold: int,
    epoch: Union[int, None],
    metrics: Dict[str, float],
    writer: Union[torch.utils.tensorboard.writer.SummaryWriter, None] = None,
    global_step: Union[int, None] = None,
) -> None:
    """Receives data about the phase, fold and epoch and prints out the metrics associated with them. Optionally, it can also save the metrics" evolution throughout training/testing to a tensorboard writer.

    Args:
        phase (str): The current phase of "train", "validation" or "test".
        fold (int): The current fold on "train" phase, or the total number of folds on 
            "test" phase.
        epoch (Union[int, None]): The current epoch on "train" phase or None on "test" phase.
        metrics (Dict[str, float]): Dictionary containing the classification metrics (accuracy, precession, recall, f-beta score) and loss. Example:
            metrics = {
                "accuracy": 0.94,
                "precession": 0.80,
                "recall": 0.69,
                "fscore": 0.76,
                "loss": 0.18,
            }
        writer (Union[torch.utils.tensorboard.writer.SummaryWriter, None], optional): Tensorboard
            SummaryWriter object. If it is None then metrics will not be saved to tensorboard
            Defaults to None.
        global_step (Union[int, None], optional): Global step that the tensorboard writer uses. 
            If either this or writer is None, it will not save the metrics on tensorboard SummaryWriter. Defaults to None.

    Raises:
        ValueError: If the phase is not one of "train", "validation" or "test".
    """
    if phase not in ["train", "validation", "test"]:
        raise ValueError(f"Phase {phase} not supported. Please choose between 'train', 'validation' and 'test'")

    # Print Metrics
    epoch_text = f"epoch: {epoch} | " if epoch is not None else ""

    print(f"{phase} || {epoch_text}", end="")

    for key, metric in metrics.items():
        print(f"{key}: {metric:.4f} | ", end="")
    print()

    # Save Metrics to Tensorboard
    if writer is not None and global_step is not None:
        for key, metric in metrics.items():
            writer.add_scalars(
                main_tag=f"{phase}_{key}",
                tag_scalar_dict={
                    f"{fold}_f": metric,
                },
                global_step=global_step
            )
        writer.close()


def _average_metrics(
    results: Dict[int, Dict[str, float]],
    num_folds: int
) -> Dict[str, float]:
    """Receives a dictionary of test results produced in each fold and produces another, which contains the average of each metric.

    Args:
        results (Dict[int, Dict[str, float]]): The dictionary of results produced by each fold. 
            Example of experiment with 2 folds containing 5 classification metrics (accuracy, precession, recall, f-score) and duration (time):
                results = {
                    0: {
                        "accuracy": 0.92,
                        "precession": 0.84,
                        "recall": 0.61,
                        "f-score": 0.78,
                        "time": 115, 
                    },
                    1: {
                        "accuracy": 0.90,
                        "precession": 0.83,
                        "recall": 0.62,
                        "f-score": 0.79,
                        "time": 117,
                    }
                }
        num_folds (int): The number of folds used in the experiment.

    Returns:
        Dict[str, float]: The dictionary of average metrics. Example of experiment with 2 folds
            containing 5 classification metrics (accuracy, precession, recall, f-score) and duration (time):
                metrics_avg = {
                    "Accuracy": 0.91,
                    "Precession": 0.84,
                    "Recall": 0.62,
                    "F-Score": 0.79,
                    "time": 116,   
                }
    """
    metrics_avg = {key: 0.0 for key in results[0]}

    for key, metrics_dict in results.items():
        print(f"fold {key} || ", end="")
        for key, metric in metrics_dict.items():
            print(f"{key}: {metric:.4f} | ", end="")
            metrics_avg[key] += metric
        print()

    print("\naverage || ", end="")

    for key, metric_sum in metrics_avg.items():
        metrics_avg[key] = metric_sum / num_folds
        print(f"{key}: {metrics_avg[key]:.4f} | ", end="")

    return metrics_avg


def k_fold_cross_validation(
    model_name: str,
    train_dataset: torchvision.datasets.ImageFolder,
    test_dataset: torchvision.datasets.ImageFolder,
    loss_fn: torch.nn.Module,
    hidden_units: int,
    device: torch.device,
    num_epochs: int,
    models_path: Path,
    experiment_name: str,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    optimizer_name: str = "Adam",
    num_folds: int = 2,
    save_models: bool = False,
    writer: Union[torch.utils.tensorboard.writer.SummaryWriter, None] = None
) -> Dict[str, float]:
    # TODO: Add docstring
    """Performs k-fold cross validation. The train_dataset is split into k folds. The k-1 folds are
    used to train the model (update parameters) and the k-th fold for validation (test the model),
    in each epoch. This process is repeated k times, so that each fold is used for validation once 
    (resulting in different train process each time). At the end of each fold, the model is 
    tested on the test_dataset. In the end, the average of the metrics produced on test_dataset is 
    returned.

    Args:
        model_name (str): Model to be used. The list of available models is: "tiny_vgg".
        train_dataset (torchvision.datasets.ImageFolder): The training dataset.
        test_dataset (torchvision.datasets.ImageFolder): The test dataset.
        loss_fn (torch.nn.Module): Loss function to be used.
        hidden_units (int): Number of hidden units to be used in the model.
        device (torch.device): Device on which the calculations will be performed ("cpu", "cuda", 
            "mps")
        num_epochs (int): Number of epochs to train the model.
        root_dir (Path): Root directory of the project.
        batch_size (int, optional): Batch size used to load the data. Defaults to 32.
        learning_rate (float, optional): Learning rate used to update the data. Defaults to 1e-3.
        optimizer_name (str, optional): Optimizer used to update the data. The list of available 
            optimizers is: "adam", "sgd". Defaults to "Adam".
        num_folds (int, optional): Number of folds to split the data into. Defaults to 2 
            (minimum number that can be used).
        save_models (bool, optional): Controls if the model's parameters are saved in each fold.
            Defaults to False.
        writer (Union[torch.utils.tensorboard.writer.SummaryWriter, None], optional): The 
            SummaryWriter object used to save metrics to tensorboard. If it is None nothing is 
            saved to tensorboard Defaults to None.

    Returns:
        Dict[str, float]: Dictionary containing the average of the classification metrics on the test_dataset. Example:
            metrics_avg = {
                "accuracy": 0.90,
                "precession": 0.88,
                "recall": 0.60,
                "fscore": 0.75,
                "time": 119,
            } 
    """
    kf = KFold(n_splits=num_folds, shuffle=True)

    results = {}

    train_indices = np.arange(len(train_dataset))

    # Define the data loader for the test set
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
    )

    for fold, (train_idx, validation_idx) in enumerate(kf.split(train_indices)):
        print(f"\nFold {fold}")
        print("-------")

        start_fold_time = timer()

        # Define the data loaders for the current fold
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_idx.tolist()),
        )
        validation_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(validation_idx.tolist()),
        )

        # Initialize the model
        model = _get_model(
            model_name=model_name,
            hidden_units=hidden_units
        ).to(device)

        # Initialize the optimizer
        optimizer = _get_optimizer(
            model=model,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
        )

        for epoch in tqdm(range(num_epochs)):
            # Train the model on the current fold
            train_metrics = _cross_validation_train(
                model=model,
                device=device,
                train_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer
            )

            _display_metrics(
                fold=fold,
                epoch=epoch,
                metrics=train_metrics,
                phase="train",
                global_step=epoch,
                writer=writer,
            )

            # Evaluate the model on the validation set
            validation_metrics = _cross_validation_test(
                model=model,
                device=device,
                test_loader=validation_loader,
                loss_fn=loss_fn,
            )

            _display_metrics(
                fold=fold,
                epoch=epoch,
                metrics=validation_metrics,
                phase="validation",
                global_step=epoch,
                writer=writer,
            )

        # Save the model for the current fold
        if save_models:
            EXTRA = f"{fold}_f_{num_epochs}_e_{batch_size}_bs_{hidden_units}_hu_{learning_rate}_lr"

            save_model(
                model=model,
                models_path=models_path,
                model_name=model_name,
                experiment_name=experiment_name,
                extra=EXTRA
            )

        # Evaluate the model on the test set
        test_metrics = _cross_validation_test(
            model=model,
            device=device,
            test_loader=test_loader,
            loss_fn=loss_fn,
        )

        _display_metrics(
            fold=num_folds,
            epoch=None,
            metrics=test_metrics,
            phase="test",
            global_step=fold,
            writer=writer,
        )

        end_fold_time = timer()

        results[fold] = test_metrics
        results[fold]["time"] = end_fold_time - start_fold_time

    # Print k-fold cross validation results
    print(f"\nK-Fold Cross Validation Results for {num_folds} Folds")
    print("--------------------------------------------")

    metrics_avg = _average_metrics(
        results=results,
        num_folds=num_folds
    )

    return metrics_avg
