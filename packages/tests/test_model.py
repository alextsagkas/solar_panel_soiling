import os
from pathlib import Path
from typing import List, Union

import torch
import torch.backends.mps

from packages.utils.configuration import models_dir, results_dir
from packages.utils.inference import inference
from packages.utils.load_data import get_dataloader
from packages.utils.models import GetModel
from packages.utils.storage import load_hyperparameters
from packages.utils.transforms import GetTransforms


def test_model(
    device: torch.device,
    test_timestamp_list: List[str],
    timestamp_list: List[str],
    save_dir: Path = models_dir,
    extra: Union[str, None] = None,
    save_images: bool = False,
    **kwargs,
) -> None:
    """Tests a model on the test set. It uses the `data/results/` directory to test the model on.
    The model is loaded using its corresponding timestamp, which is the `test_timestamp_list`. 

    Also, despite the evaluation of classification metrics, images are saved in `debug/test_model/
    YYYY-MM-DD, HH-MM-SS/` containing the predicted class and the ground truth, while displaying the
    probability with which the choice was made. This behavior is controlled optionally by the
    `save_images` argument.

    Via the `**kwargs` argument, the test transform, that is picked by the train_transform name of 
    the loaded parameters, can be configured further. An example is the following:
        kwargs = {
            "test_transform_config": {
                "resize_size": 256,
                "crop_size": 224,
        }

    Args:
        device (torch.device): Device to use for the testing.
        test_timestamp_list (List[str]): List of timestamp (YYYY-MM-DD, HH-MM-SS) the trained
            model used.
        timestamp_list (List[str]): List of timestamp (YYYY-MM-DD, HH-MM-SS) the test_model was
            called.
        save_dir (Path, optional): Directory where the model is saved. Defaults to models_dir. It 
            can be used to load checkpoints from training.
        extra (Union[str, None], optional): Extra string to append to the model name. Used to 
            choose which epoch of checkpoint you would like to pick. Defaults to None.
        save_images (bool, optional): Whether to save images or not. Defaults to False.
    """
    test_transform_config = kwargs.get("test_transform_config", None)

    # Load test hyperparameters
    test_hyperparameters = load_hyperparameters(
        test_timestamp_list=test_timestamp_list,
    )

    model_name = test_hyperparameters["model_name"]
    model_config = test_hyperparameters.get("model_config", None)
    loaded_timestamp_list = test_hyperparameters["timestamp_list"]
    train_transform_name = test_hyperparameters["train_transform_name"]

    # Load model
    if extra is not None:
        MODEL_SAVE_PATH = (
            save_dir /
            loaded_timestamp_list[0] /  # type: ignore
            f"{loaded_timestamp_list[1]}_{extra}.pth")  # type: ignore
    else:
        MODEL_SAVE_PATH = (
            save_dir /
            loaded_timestamp_list[0] /  # type: ignore
            f"{loaded_timestamp_list[1]}.pth")  # type: ignore

    print(f"[INFO] Model loaded from {MODEL_SAVE_PATH}")

    model = GetModel(
        model_name=model_name,  # type: ignore
        config=model_config,  # type: ignore
    ).get_model().to(device)
    model.load_state_dict(torch.load(f=str(MODEL_SAVE_PATH)))

    # Load transforms
    transform_obj = GetTransforms(
        test_transform_name=train_transform_name,  # type: ignore
        test_config=test_transform_config,
    )

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
        save_images=save_images,
    )
