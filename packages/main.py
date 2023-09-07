from datetime import datetime

import torch
import torch.backends.mps

from packages.tests.test_data import test_transform
from packages.tests.test_model import test_model
from packages.tests.test_resume import test_resume
from packages.tests.test_scraping import test_scraping
from packages.tests.test_solver import test_solver
from packages.utils.configuration import checkpoint_dir
from packages.utils.storage import save_hyperparameters

if __name__ == "__main__":

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Setup hyperparameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    test_names = ["test_solvers-simple", "test_model", "test_resume", "test_data", "test_scraping"]
    test_name = test_names[4]

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
            "model_name": "resnet50",
            "num_epochs": 10,
            "batch_size": 32,
            "optimizer_name": "adam",
            "optimizer_config": {
                "learning_rate": 1e-3,
                "weight_decay": 1e-3
            },
            "scheduler_name": "steplr",
            "scheduler_config": {
                "step_size": 3,
                "gamma": 0.1,
            },
            "train_transform_name": "resnet",
            "train_transform_config": {
                "random_rotation": 180,
                "num_magnitude_bins": 31,
            },
            "test_transform_name": "resnet",
            "timestamp_list": timestamp_list,
            "device": device,
        }
        save_hyperparameters(hyperparameters=hyperparameters)
        test_solver(**hyperparameters)
    elif test_name == "test_model":
        hyperparameters = {
            "test_name": test_name,
            "device": device,
            "save_dir": checkpoint_dir,
            "extra": "epoch_8",
            "test_timestamp_list": ["2023-09-07", "13-45-48"],
            "timestamp_list": timestamp_list,
            "save_images": False,
        }
        save_hyperparameters(hyperparameters=hyperparameters)
        test_model(**hyperparameters)
    elif test_name == "test_resume":
        hyperparameters = {
            "load_config": {
                "checkpoint_timestamp_list": ["2023-09-06", "23-47-28"],
                "load_epoch": 9,
            },
            "test_name": test_name,
            "model_name": "resnet34",
            "num_epochs": 7,
            "batch_size": 32,
            "optimizer_name": "adam",
            "optimizer_config": {
                "learning_rate": 5e-5,
                "weight_decay": 1e-4
            },
            "scheduler_name": "steplr",
            "scheduler_config": {
                "step_size": 2,
                "gamma": 0.1,
            },
            "train_transform_name": "resnet",
            "train_transform_config": {"random_rotation": 180},
            "test_transform_name": "resnet",
            "timestamp_list": timestamp_list,
            "device": device,
        }
        save_hyperparameters(hyperparameters=hyperparameters)
        test_resume(**hyperparameters)
    elif test_name == "test_data":
        hyperparameters = {
            "timestamp_list": timestamp_list,
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
            "max_images": 50,
            "delay": 0,
            "url": "https://www.google.com/search?client=safari&sca_esv=563382129&rls=en&sxsrf=AB5stBjcBYJvBNXbZkQoxcLP8mUEu8OMlQ:1694090812474&q=solar+panel+images&tbm=isch&source=lnms&sa=X&ved=2ahUKEwjN0OCIxJiBAxXvV0EAHalWC-AQ0pQJegQICBAB&biw=1016&bih=1175&dpr=2",
        }
        save_hyperparameters(hyperparameters=hyperparameters)
        test_scraping(**hyperparameters)
