import os
from timeit import default_timer as timer
from typing import Dict, List, Union

import numpy as np
import torch
import torch.utils.data
import torch.utils.tensorboard
import torchvision
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassFBetaScore,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm import tqdm
from typing_extensions import Self

from packages.utils.configuration import checkpoint_dir, tensorboard_dir
from packages.utils.models import GetModel
from packages.utils.optim import GetOptimizer
from packages.utils.storage import save_metrics, save_model


class Solver:
    """Class that trains and tests a Pytorch model. It provides functionality for a simple train 
    function and a k-fold cross validation train function. 

    In the first case, the model is trained on the train dataset for num_epochs and evaluated on 
    the test dataset in the end. The results on the test dataset are saved in the debug/metrics.
    Also, the metrics generated on every epoch are saved on tensorboard, in debug/runs folder.

    In the second case, the train dataset is split into num_folds folds and the model is trained on 
    the 1 - mun_folds folds for num_epochs, whereas it is evaluated in the 1 fold remaining
    (validation set), for every epoch. The process is repeated until every single fold has been 
    used as a validation set. In the end of each fold the model is also evaluated on the test set. 
    The results on the test set are averaged out and saved in the debug/metrics. Also, the metrics
    generated on each epoch are saved on tensorboard in debug/runs folder.

    In addition, any model that is trained is saved in the models/ folder.

    Attributes:
        model_obj (GetModel): The model object to be trained and tested.
        num_folds (int): Number of folds to be used in k-fold cross validation.
        num_epochs (int): Number of epochs to train the model.
        batch_size (int): Number of samples per batch.
        loss_fn (torch.nn.Module): Loss function to be used.
        optimizer_name (str): String that identifies the optimizer to be used.
        scheduler_name (Union[str, None]): String that identifies the scheduler to be used.
        device (torch.device): Device to be used to load the model.
        train_dataset (torchvision.datasets.ImageFolder): Train dataset.
        test_dataset (torchvision.datasets.ImageFolder): Test dataset.
        timestamp_list (List[str]): List of strings that contain the timestamp of the experiment.
        optimizer_config (Union[Dict[str, float], None]): Dictionary with the configuration of the
            optimizer.
        scheduler_config (Union[Dict, None]): Dictionary with the configuration of the scheduler.

    Methods:
        _train_step: Trains a PyTorch model for one epoch.
        _test_step: Tests a PyTorch model for one epoch.
        _create_writer: Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to
            the directory: debug/runs/YYYY-MM-DD/HH-MM-SS/".
        _display_metrics: Receives data about the phase and epoch and prints out the metrics while
            saving them to a tensorboard writer.
        train_model: Trains and tests a PyTorch model.
        _average_metrics: Receives a dictionary of test results produced in each fold and produces
            another, which contains the average of each metric.
        train_model_kfold: Trains and tests a model using k-fold cross validation.
    """

    def __init__(
        self: Self,
        model_obj: GetModel,
        num_epochs: int,
        batch_size: int,
        loss_fn: torch.nn.Module,
        optimizer_name: str,
        device: torch.device,
        train_dataset: torchvision.datasets.ImageFolder,
        test_dataset: torchvision.datasets.ImageFolder,
        timestamp_list: List[str],
        num_folds: Union[int, None] = None,
        scheduler_name: Union[str, None] = None,
        **kwargs,
    ) -> None:
        """Initializes the Solver class.

        Args:
            model_obj (GetModel): The model object to be trained and tested.
            num_epochs (int): number of epochs to train the model.
            batch_size (int): number of samples per batch.
            loss_fn (torch.nn.module): Loss function to be used.
            optimizer_name (str): String that identifies the optimizer to be used.
            device (torch.device): device to be used to load the model.
            train_dataset (torchvision.datasets.ImageFolder): Train dataset.
            test_dataset (torchvision.datasets.ImageFolder): Test dataset.
            timestamp_list (List[str]): List of strings that contain the timestamp of the experiment.
                Used for storage path.
            num_folds (Union[int, None], optional): Number of folds to be used in k-fold cross  
                validation.
            scheduler_name (Union[str, None], optional): String that identifies the scheduler to be
                used. Defaults to None.
            kwargs (dict): Dictionary of optional arguments. Defaults to None.
        """
        self.model_obj = model_obj

        self.num_folds = num_folds
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.loss_fn = loss_fn
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name

        self.device = device

        # Datasets
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # Extra
        self.timestamp_list = timestamp_list

        # Configuration
        self.optimizer_config = kwargs.get("optimizer_config", None)
        self.scheduler_config = kwargs.get("scheduler_config", None)

    def _train_step(
        self: Self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """Trains a PyTorch model for one epoch.

        Args:
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): Optimizer to be used.
            dataloader (torch.utils.data.DataLoader): Train DataLoader.

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
            "accuracy": MulticlassAccuracy(num_classes=2).to(self.device),
            "precession": MulticlassPrecision(num_classes=2).to(self.device),
            "recall": MulticlassRecall(num_classes=2).to(self.device),
            "fscore": MulticlassFBetaScore(num_classes=2, beta=2.0).to(self.device),
        }

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            y_pred = model(X)

            loss = self.loss_fn(y_pred, y)
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
        self: Self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """Tests a PyTorch model for one epoch.

        Args:
            model (torch.nn.Module): The model to be tested.
            dataloader (torch.utils.data.DataLoader): Test DataLoader.

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
            "accuracy": MulticlassAccuracy(num_classes=2).to(self.device),
            "precession": MulticlassPrecision(num_classes=2).to(self.device),
            "recall": MulticlassRecall(num_classes=2).to(self.device),
            "fscore": MulticlassFBetaScore(num_classes=2, beta=2.0).to(self.device),
        }

        with torch.inference_mode():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                test_pred_logits = model(X)

                test_loss += self.loss_fn(test_pred_logits, y)

                test_pred_labels = test_pred_logits.argmax(dim=1)

                for key, _ in test_metrics.items():
                    test_metrics[key].update(test_pred_labels, y)

        test_loss = test_loss / len(dataloader)

        for key, _ in test_metrics.items():
            test_metrics[key] = test_metrics[key].compute().item()
        test_metrics["loss"] = test_loss

        return test_metrics

    def _create_writer(
        self: Self,
        timestamp_list: List[str],
    ) -> torch.utils.tensorboard.writer.SummaryWriter:
        """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to the 
        directory: debug/runs/YYYY-MM-DD/HH-MM-SS/"

        Args:
            timestamp_list (List[str]): List of strings that contain the timestamp of the 
            experiment. Form: [YYYY-MM-DD, HH-MM-SS].

        Returns:
            torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.
        """

        # Create tensorboard directory if it doesn't exist
        tensorboard_dir.mkdir(exist_ok=True, parents=True)

        log_dir = os.path.join(tensorboard_dir, *timestamp_list)

        print(f"[INFO] Created SummaryWriter, saving to: {log_dir}")

        return SummaryWriter(log_dir=log_dir)

    def _display_metrics(
        self: Self,
        phase: str,
        epoch: Union[int, None],
        metrics: Dict[str, float],
        writer: torch.utils.tensorboard.writer.SummaryWriter,
        global_step: int,
        fold: Union[int, None] = None,
    ) -> None:
        """Receives data about the phase and epoch and prints out the metrics associated with them. 
        It also saves the metrics' evolution throughout training/testing to a tensorboard writer.

        Args:
            phase (str): The current phase of "train" or "test".
            epoch (Union[int, None]): The current epoch. If it is None then it is a test phase 
                of the k fold train.
            metrics (Dict[str, float]): Dictionary containing the classification metrics and loss.
            writer (torch.utils.tensorboard.writer.SummaryWriter):  
                Tensorboard SummaryWriter object. If it is None then metrics will not be saved 
                to tensorboard.
            global_step (int): The current global step. When simple train is used, it is the same
                as the epoch.

        Raises:
            ValueError: If the phase is not one of "train" or "test".
        """
        supported_phases = ["train", "validation", "test"]
        if phase not in supported_phases:
            raise ValueError(
                f"Phase {phase} not supported. Please choose between {supported_phases}"
            )

        # Print Metrics
        epoch_text = f"epoch: {epoch} | " if epoch is not None else ""

        print(f"{phase} || {epoch_text}", end="")

        for key, metric in metrics.items():
            print(f"{key}: {metric:.4f} | ", end="")
        print()

        # Save Metrics to Tensorboard
        for key, metric in metrics.items():

            main_tag = "classification_metrics" if key != "loss" else "loss"
            scalar_name = f"{phase}_{key}_{fold}_f" if fold is not None else f"{phase}_{key}"

            writer.add_scalars(
                main_tag=main_tag,
                tag_scalar_dict={
                    scalar_name: metric,
                },
                global_step=global_step,
            )

            writer.close()

    def _save_checkpoint(
        self: Self,
        model: torch.nn.Module,
        extra: str,
    ) -> None:
        """Save checkpoint of the model training process so as to be able to resume it later. The
        checkpoint is saved in the checkpoint_dir/YYYY-MM-DD/HH-MM-SS_{extra}.pth. 

        Args:
            self (Self): Instance of the Solver class.
            model (torch.nn.Module): The model to be saved.
            extra (str): Extra information concerning the training (used as name of file saved).
        """
        save_model(
            model=model,
            timestamp_list=self.timestamp_list,
            save_dir=checkpoint_dir,
            extra=extra,
        )

    def train_model(
        self: Self,
    ) -> Dict[str, float]:
        """Trains and tests a PyTorch model.

        Passes a target PyTorch models through _train_step() function for a number of epochs.
        When the training is done, passes the same updated model through _test_step() function,
        to evaluate the model's performance on the test dataset.

        Calculates, prints and stores evaluation metrics on training set throughout.

        Stores metrics to specified writer in debug/runs/ folder and in the debug/metrics folder.

        Returns:
            Dict[str, float]: A dictionary containing the classification metrics, the loss for
            the test dataset, and the time taken to train it. Example:
                metrics = {
                    "accuracy": 0.94,
                    "precession": 0.80,
                    "recall": 0.69,
                    "fscore": 0.76,
                    "loss": 0.18,
                    "time": 0.18,
                }
        """
        # Count model time
        start_time = timer()

        # Dataloaders
        NUM_CORES = os.cpu_count()

        print(f"[INFO] Using {NUM_CORES} workers for data loading.")

        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=NUM_CORES if NUM_CORES is not None else 1,
        )
        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NUM_CORES if NUM_CORES is not None else 1,
        )

        # Create writer
        writer = self._create_writer(timestamp_list=self.timestamp_list)

        # Create model
        model = self.model_obj.get_model().to(self.device)

        print(f"[INFO] Using {self.device} for training and testing.")

        # Create Optimizer & Scheduler
        optimizer_obj = GetOptimizer(
            optimizer_name=self.optimizer_name,
            params=model.parameters(),
            config=self.optimizer_config,
            scheduler_name=self.scheduler_name,
            scheduler_config=self.scheduler_config,
        )

        optimizer = optimizer_obj.get_optimizer()

        if self.scheduler_name is not None:
            scheduler = optimizer_obj.get_scheduler()
        else:
            scheduler = None

        # Initialize metrics to avoid type error after the loop
        test_metrics = {}
        best_test_metrics = {
            "accuracy": 0.0,
        }

        for epoch in tqdm(range(self.num_epochs)):
            # Optimization on train set
            train_metrics = self._train_step(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
            )
            self._display_metrics(
                phase="train",
                epoch=epoch,
                metrics=train_metrics,
                writer=writer,
                global_step=epoch,
            )

            # Evaluation on test set
            test_metrics = self._test_step(
                model=model,
                dataloader=test_dataloader,
            )
            self._display_metrics(
                phase="test",
                epoch=epoch,
                metrics=test_metrics,
                writer=writer,
                global_step=epoch,
            )
            if test_metrics["accuracy"] > best_test_metrics["accuracy"]:
                best_test_metrics = test_metrics
                best_test_metrics["epoch"] = epoch

            self._save_checkpoint(
                model=model,
                extra=f"epoch_{epoch}",
            )
            optimizer_obj.update_scheduler(
                test_metrics=test_metrics,
                scheduler=scheduler,
            )

        save_model(
            model=model,
            timestamp_list=self.timestamp_list,
        )

        end_time = timer()

        metrics = best_test_metrics
        metrics["time"] = end_time - start_time

        save_metrics(
            metrics=metrics,
            timestamp_list=self.timestamp_list,
        )

        return metrics

    def _average_metrics(
        self: Self,
        results: Dict[int, Dict[str, float]],
    ) -> Dict[str, float]:
        """Receives a dictionary of test results produced in each fold and produces another, which 
        contains the average of each metric.

        Args:
            results (Dict[int, Dict[str, float]]): The dictionary of results produced by each fold. 
                Example of experiment with 2 folds containing 5 classification metrics (accuracy, 
                precession, recall, f-score) and duration (time):
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

        Returns:
            Dict[str, float]: The dictionary of average metrics. Example of experiment with 2 folds
                containing 5 classification metrics (accuracy, precession, recall, f-score) and 
                duration (time):
                    metrics_avg = {
                        "Accuracy": 0.91,
                        "Precession": 0.84,
                        "Recall": 0.62,
                        "F-Score": 0.79,
                        "time": 116,   
                    }
        """
        if self.num_folds is None:
            raise ValueError("num_folds must be specified")

        metrics_avg = {key: 0.0 for key in results[0]}

        for key, metrics_dict in results.items():
            print(f"fold {key} || ", end="")
            for key, metric in metrics_dict.items():
                print(f"{key}: {metric:.4f} | ", end="")
                metrics_avg[key] += metric
            print()

        print("\naverage || ", end="")

        for key, metric_sum in metrics_avg.items():
            metrics_avg[key] = metric_sum / self.num_folds
            print(f"{key}: {metrics_avg[key]:.4f} | ", end="")
        print()

        return metrics_avg

    def train_model_kfold(
        self: Self,
    ) -> Dict[str, float]:
        """Trains and test a model using k-fold cross validation.

        Passes a target PyTorch models through _train_step() function for a number of epochs.
        When the training is done, passes the same updated model through _test_step() function,
        to evaluate the model's performance on the test dataset.

        Calculates, prints and stores evaluation metrics on training set throughout.

        Stores metrics to specified writer in debug/runs/folder and in the debug/metrics folder.

        Returns:
            Dict[str, float]: A dictionary containing the classification metrics, the loss for
            the test dataset, and the time taken to train it. Example:
                metrics = {
                    "accuracy": 0.94,
                    "precession": 0.80,
                    "recall": 0.69,
                    "fscore": 0.76,
                    "loss": 0.18,
                    "time": 0.18,
                }
        """
        if self.num_folds is None:
            raise ValueError("num_folds must be specified")

        # Split data into k folds by shuffling the indices
        kf = KFold(n_splits=self.num_folds, shuffle=True)

        results = {}

        train_indices = np.arange(len(self.train_dataset))

        # Define the data loader for the test set
        NUM_CORES = os.cpu_count()

        print(f"[INFO] Using {NUM_CORES} workers for data loading.")

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NUM_CORES if NUM_CORES is not None else 1,
        )

        # Create writer
        writer = self._create_writer(timestamp_list=self.timestamp_list)

        for fold, (train_idx, validation_idx) in enumerate(kf.split(train_indices)):
            print(f"\nFold {fold}")
            print("-------")

            start_fold_time = timer()

            # Define the data loaders for the current fold
            train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                num_workers=NUM_CORES if NUM_CORES is not None else 1,
                sampler=SubsetRandomSampler(train_idx.tolist()),
            )
            validation_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                num_workers=NUM_CORES if NUM_CORES is not None else 1,
                sampler=SubsetRandomSampler(validation_idx.tolist()),
            )

            # Initialize the model
            model = self.model_obj.get_model().to(self.device)

            print(f"[INFO] Using {self.device} for training, validation and testing.")

            # Initialize the optimizer
            optimizer = GetOptimizer(
                params=model.parameters(),
                optimizer_name=self.optimizer_name,
                config=self.optimizer_config,
            ).get_optimizer()

            for epoch in tqdm(range(self.num_epochs)):
                # Train the model on the current fold
                train_metrics = self._train_step(
                    model=model,
                    optimizer=optimizer,
                    dataloader=train_loader,
                )
                self._display_metrics(
                    fold=fold,
                    epoch=epoch,
                    metrics=train_metrics,
                    phase="train",
                    global_step=epoch,
                    writer=writer,
                )

                # Evaluate the model on the validation set
                validation_metrics = self._test_step(
                    model=model,
                    dataloader=validation_loader,
                )

                self._display_metrics(
                    fold=fold,
                    epoch=epoch,
                    metrics=validation_metrics,
                    phase="validation",
                    global_step=epoch,
                    writer=writer,
                )

            # Save the model for the current fold
            EXTRA = f"{fold}_f"

            save_model(
                model=model,
                timestamp_list=self.timestamp_list,
                extra=EXTRA,
            )

            # Evaluate the model on the test set
            test_metrics = self._test_step(
                model=model,
                dataloader=test_loader,
            )

            self._display_metrics(
                fold=self.num_folds,
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
        print(f"\nK-Fold Cross Validation Results for {self.num_folds} Folds")
        print("--------------------------------------------")

        metrics_avg = self._average_metrics(
            results=results,
        )

        save_metrics(
            metrics=metrics_avg,
            timestamp_list=self.timestamp_list,
        )

        return metrics_avg
