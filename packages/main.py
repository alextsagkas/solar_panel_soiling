from datetime import datetime

import torch
import torch.backends.mps

from packages.tests.test_data import test_transform
from packages.tests.test_model import test_model
from packages.tests.test_solver import test_kfold_solver, test_solver
from packages.utils.storage import save_hyperparameters

if __name__ == "__main__":

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Setup hyperparameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    test_names = ["test_solvers-simple", "test_solvers-kfold", "test_model", "test_data"]
    model_names = ["tiny_vgg", "tiny_vgg_dropout"]

    timestamp_list = datetime.now().strftime("%Y-%m-%d_%H-%M-%S").split("_")

    test_name = test_names[3]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Setup Device ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple GPU
    else:
        device = torch.device("cpu")  # CPU (if others unavailable)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Run Test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    print(f"[INFO] Running {test_name} test.")

    if test_name == "test_solvers-simple":
        hyperparameters = {
            "test_name": test_name,
            "model_name": model_names[1],
            "model_config": {"hidden_units": 64,
                             "dropout_rate": 0.5},
            "num_epochs": 1,
            "batch_size": 32,
            "optimizer_name": "adam",
            "transform_name": "trivial",
            "timestamp_list": timestamp_list
        }

        save_hyperparameters(hyperparameters=hyperparameters)

        test_solver(
            model_name=hyperparameters["model_name"],
            num_epochs=hyperparameters["num_epochs"],
            batch_size=hyperparameters["batch_size"],
            optimizer_name=hyperparameters["optimizer_name"],
            transform_name=hyperparameters["transform_name"],
            timestamp_list=hyperparameters["timestamp_list"],
            device=device,
            model_config=hyperparameters["model_config"]
        )
    elif test_name == "test_solvers-kfold":
        hyperparameters = {
            "test_name": test_name,
            "model_name": model_names[0],
            "model_config": {"hidden_units": 64},
            "num_folds": 3,
            "num_epochs": 2,
            "batch_size": 32,
            "optimizer_name": "adam",
            "transform_name": "trivial",
            "timestamp_list": timestamp_list
        }

        save_hyperparameters(hyperparameters=hyperparameters)

        test_kfold_solver(
            model_name=hyperparameters["model_name"],
            num_folds=hyperparameters["num_folds"],
            num_epochs=hyperparameters["num_epochs"],
            batch_size=hyperparameters["batch_size"],
            optimizer_name=hyperparameters["optimizer_name"],
            transform_name=hyperparameters["transform_name"],
            timestamp_list=hyperparameters["timestamp_list"],
            device=device,
        )
    elif test_name == "test_model":
        hyperparameters = {
            "test_name": test_name,
            "timestamp_list": timestamp_list,
            "test_timestamp_list": ["2023-09-05", "13-19-06"],
        }

        save_hyperparameters(hyperparameters=hyperparameters)

        test_model(
            device=device,
            test_timestamp_list=hyperparameters["test_timestamp_list"],
            timestamp_list=hyperparameters["timestamp_list"],
        )
    elif test_name == "test_data":
        hyperparameters = {
            "test_name": test_name,
            "transform_name": "trivial",
            "transform_config": {"resize_size": 128},
            "timestamp_list": timestamp_list,
        }

        save_hyperparameters(hyperparameters=hyperparameters)

        test_transform(
            transform_name=hyperparameters["transform_name"],
            transform_config=hyperparameters["transform_config"],
        )
