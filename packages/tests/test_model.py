import ast
import os
from typing import List

import torch
import torch.backends.mps

from packages.utils.configuration import models_dir, results_dir, test_model_dir
from packages.utils.inference import inference
from packages.utils.load_data import get_dataloader
from packages.utils.models import GetModel
from packages.utils.storage import load_hyperparameters
from packages.utils.transforms import GetTransforms


def test_model(
    device: torch.device,
    test_timestamp_list: List[str],
    timestamp_list: List[str],
) -> None:
    """Tests a model on the test set.

    Args:
        device (torch.device): Device to use for the testing.
        test_timestamp_list (List[str]): List of timestamp (YYYY-MM-DD, HH-MM-SS) the trained
            model used.
        timestamp_list (List[str]): List of timestamp (YYYY-MM-DD, HH-MM-SS) the test_model was
            called.
    """
    # Load test hyperparameters
    test_hyperparameters = load_hyperparameters(
        test_timestamp_list=test_timestamp_list,
    )

    model_name = test_hyperparameters["model_name"]
    loaded_timestamp_list = test_hyperparameters["timestamp_list"]
    transform_name = test_hyperparameters["transform_name"]

    # Load model
    MODEL_SAVE_PATH = models_dir / loaded_timestamp_list[0] / f"{loaded_timestamp_list[1]}.pth"

    print(f"[INFO] Model loaded from {MODEL_SAVE_PATH}")

    model = GetModel(model_name=model_name).get_model().to(device)
    model.load_state_dict(torch.load(f=str(MODEL_SAVE_PATH)))

    # Load transforms
    transform_obj = GetTransforms(transform_name=transform_name)

    # Load data
    BATCH_SIZE = 1

    NUM_WORKERS = os.cpu_count()

    test_dataloader, class_names = get_dataloader(
        dir=str(results_dir),
        data_transform=transform_obj.get_test_transform(),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS if NUM_WORKERS is not None else 1,
        shuffle=False
    )

    # Inference
    inference(
        model=model,
        test_dataloader=test_dataloader,
        class_names=class_names,
        device=device,
        timestamp_list=timestamp_list,
    )
