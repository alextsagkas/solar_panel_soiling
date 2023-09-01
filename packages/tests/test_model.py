import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.backends.mps

from packages.models.tiny_vgg import TinyVGG
from packages.utils.inference import inference
from packages.utils.load_data import get_dataloader
from packages.utils.transforms import test_data_transform


def test_model(
    test_dir: Path,
    num_fold: int,
    num_epochs: int,
    batch_size: int,
    hidden_units: int,
    learning_rate: float,
    device: torch.device,
    model_name: str,
    models_path: Path,
    num_workers: int,
    test_model_path: Path,
) -> Tuple[Dict[str, float], str, str]:
    """Tests a model on the test set.

    Args:
        test_dir (Path): Test set directory.
        num_fold (int): Number of the fold (-1 if not k-fold).
        num_epochs (int): Number of epochs.
        batch_size (int): Batch size.
        hidden_units (int): Number of hidden units.
        learning_rate (float): Learning rate.
        device (torch.device): Device to use ("cpu", "cuda", "mps").
        model_name (str): Name of the model.
        models_path (Path): Path to the models.
        num_workers (int): Number of workers.
        test_model_path (Path): Path where to save the results of the test.

    Returns:
        Tuple[Dict[str, float], str, str]: Dictionary with the classification metrics of the   
          experiment (accuracy, precision, recall, f1-score, loss), the extra information 
          concerning the training and the experiment name
    """
    EXPERIMENT_NAME = "test_model"

    # Load model
    if num_fold == -1:
        EXTRA = f"{num_epochs}_e_{batch_size}_bs_{hidden_units}_hu_{learning_rate}_lr"
        EXPERIMENT_DONE = "test_train"
    else:
        EXTRA = f"{num_fold}_f_{num_epochs}_e_{batch_size}_bs_{hidden_units}_hu_{learning_rate}_lr"
        EXPERIMENT_DONE = "test_kfold"
    MODEL_SAVE_DIR = models_path / model_name / EXPERIMENT_DONE
    MODEL_SAVE_NAME = EXTRA + ".pth"

    model = TinyVGG(
        input_shape=3,
        hidden_units=hidden_units,
        output_shape=2
    ).to(device)
    model.load_state_dict(torch.load(f=str(MODEL_SAVE_DIR / MODEL_SAVE_NAME)))

    print(f"[INFO] Model loaded from {MODEL_SAVE_DIR / MODEL_SAVE_NAME}")

    # Load data
    BATCH_SIZE = 1

    test_dataloader, class_names = get_dataloader(
        dir=str(test_dir),
        data_transform=test_data_transform,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        shuffle=False
    )

    # Inference
    results_metrics = inference(
        model=model,
        test_dataloader=test_dataloader,
        class_names=class_names,
        test_model_path=test_model_path,
        model_name=model_name,
        experiment_name=EXPERIMENT_NAME,
        extra=EXTRA,
        device=device
    )

    return results_metrics, EXTRA, EXPERIMENT_NAME
