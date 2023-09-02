import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.backends.mps

from packages.utils.inference import inference
from packages.utils.load_data import get_dataloader
from packages.utils.models import GetModel
from packages.utils.transforms import GetTransforms


def test_model(
    model_obj: GetModel,
    test_dir: Path,
    num_fold: int,
    num_epochs: int,
    batch_size: int,
    models_path: Path,
    num_workers: int,
    test_model_path: Path,
    transform_obj: GetTransforms,
) -> Tuple[Dict[str, float], str, str]:
    """Tests a model on the test set.

    Args:
        model_obj (GetModel): Model object to use.j
        test_dir (Path): Test set directory.
        num_fold (int): Number of the fold (-1 if not k-fold).
        num_epochs (int): Number of epochs.
        batch_size (int): Batch size.
        models_path (Path): Path to the models.
        num_workers (int): Number of workers.
        test_model_path (Path): Path where to save the results of the test.
        transform_obj (GetTransforms): Transform object to use for the data.

    Returns:
        Tuple[Dict[str, float], str, str]: Dictionary with the classification metrics of the   
          experiment (accuracy, precision, recall, f1-score, loss), the extra information 
          concerning the training and the experiment's name done (test_train or test_kfold) 
          so as to produce the model that was loaded to evaluate the data.
    """
    # Load model
    if num_fold == -1:
        EXTRA = f"{num_epochs}_e_{batch_size}_bs"
        EXPERIMENT_DONE = "test_train"
    else:
        EXTRA = f"{num_fold-1}_f_{num_epochs}_e_{batch_size}_bs"
        EXPERIMENT_DONE = "test_kfold"
    MODEL_SAVE_DIR = models_path / model_obj.model_name / EXPERIMENT_DONE / transform_obj.transform_name
    MODEL_SAVE_NAME = EXTRA + ".pth"

    model = model_obj.get_model()
    model.load_state_dict(torch.load(f=str(MODEL_SAVE_DIR / MODEL_SAVE_NAME)))

    print(f"[INFO] Model loaded from {MODEL_SAVE_DIR / MODEL_SAVE_NAME}")

    # Load data
    BATCH_SIZE = 1

    test_dataloader, class_names = get_dataloader(
        dir=str(test_dir),
        data_transform=transform_obj.get_test_transform(),
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
        model_name=model_obj.model_name,
        experiment_name=EXPERIMENT_DONE,
        transform_name=transform_obj.transform_name,
        extra=EXTRA,
        device=model_obj.device,
    )

    return results_metrics, EXTRA, EXPERIMENT_DONE
