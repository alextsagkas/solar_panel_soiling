from datetime import datetime

import torch
import torch.backends.mps

from packages.tests.test_data import test_transform
from packages.tests.test_model import test_model
from packages.tests.test_resume import test_resume
from packages.tests.test_scraping import test_scraping
from packages.tests.test_solver import test_solver
from packages.utils.configuration import (
    checkpoint_dir,
    download_test_dir,
    download_train_dir,
)
from packages.utils.storage import save_hyperparameters

if __name__ == "__main__":

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Setup hyperparameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    test_names = ["test_solvers-simple", "test_model", "test_resume", "test_data", "test_scraping"]
    test_name = test_names[0]

    timestamp_list = datetime.now().strftime("%Y-%m-%d_%H-%M-%S").split("_")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Setup Device ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple GPU
    else:
        device = torch.device("cpu")  # CPU (if others unavailable)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Run Test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    print(f"[INFO] Running {test_name} test.")

    if test_name == "test_solvers-simple":
        hyperparameters = {
            "test_name": test_name,
            "model_name": "efficientnet_v2_m",
            "num_epochs": 30,
            "batch_size": 256,
            "optimizer_name": "sgd",
            "optimizer_config": {
                "learning_rate": 1e-1,
                "momentum": 0.9,
                "weight_decay": 1e-4
            },
            "scheduler_name": "steplr",
            "scheduler_config": {
                "step_size": 10,
                "gamma": 0.1,
            },
            "train_dir": download_train_dir,
            "train_transform_name": "efficientnet",
            "train_transform_config": {
                "random_rotation": 0,
            },
            "test_dir": download_test_dir,
            "test_transform_name": "efficientnet",
            "timestamp_list": timestamp_list,
            "device": device,
        }
        save_hyperparameters(hyperparameters=hyperparameters)
        test_solver(**hyperparameters)
    elif test_name == "test_model":
        hyperparameters = {
            "test_name": test_name,
            "device": device,
            "test_dir": download_test_dir,
            "save_dir": checkpoint_dir,
            "extra": "epoch_26",
            "test_timestamp_list": ["2023-09-10", "22-51-34"],
            "timestamp_list": timestamp_list,
            "save_images": False,
        }
        save_hyperparameters(hyperparameters=hyperparameters)
        test_model(**hyperparameters)
    elif test_name == "test_resume":
        hyperparameters = {
            "load_config": {
                "checkpoint_timestamp_list": ["2023-09-10", "22-51-34"],
                "load_epoch": 26,
            },
            "test_name": test_name,
            "model_name": "efficientnet_b6",
            "num_epochs": 15,
            "batch_size": 256,
            "optimizer_name": "adam",
            "optimizer_config": {
                "learning_rate": 1e-2,
                "weight_decay": 1e-3
            },
            "scheduler_name": "steplr",
            "scheduler_config": {
                "step_size": 8,
                "gamma": 0.1,
            },
            "train_dir": download_train_dir,
            "train_transform_name": "efficientnet",
            "train_transform_config": {
                "random_rotation": 0
            },
            "test_dir": download_test_dir,
            "test_transform_name": "efficientnet",
            "timestamp_list": timestamp_list,
            "device": device,
        }
        save_hyperparameters(hyperparameters=hyperparameters)
        test_resume(**hyperparameters)
    elif test_name == "test_data":
        hyperparameters = {
            "timestamp_list": timestamp_list,
            "n": 20,
            "test_name": test_name,
            "train_transform_name": "resnet",
            "train_transform_config": {
                "random_rotation": 180,
                "num_magnitude_bins": 31,
            },
        }
        save_hyperparameters(hyperparameters=hyperparameters)
        test_transform(**hyperparameters)
    elif test_name == "test_scraping":
        hyperparameters = {
            "test_name": test_name,
            "timestamp_list": timestamp_list,
            "max_images": 300,
            "delay": 1.5,
            "url": "https://www.google.gr/search?q=solar%20panels%20images%20leaves&tbm=isch&hl=el&tbs=rimg:CXkYJ-o92B8gYQGwBcG10kwVsgIRCgIIABAAOgQIABABVSx8SD_1AAgDYAgDgAgA&sa=X&ved=0CBoQuIIBahcKEwiwpNPB6Z2BAxUAAAAAHQAAAAAQBg&biw=2032&bih=1175",
        }
        save_hyperparameters(hyperparameters=hyperparameters)
        test_scraping(**hyperparameters)
