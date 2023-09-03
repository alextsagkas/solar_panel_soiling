import os
from datetime import datetime
from pathlib import Path

import torch
import torch.backends.mps

from packages.tests.test_data import test_transform
from packages.tests.test_solver import test_kfold_solver, test_solver
from packages.utils.configuration import config_dir, data_dir, test_dir, train_dir
from packages.utils.models import GetModel
from packages.utils.transforms import GetTransforms

if __name__ == "__main__":

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Setup hyperparameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    test_names = ["test_data", "test_model", "test_solvers-simple", "test_solvers-kfold"]
    model_names = ["tiny_vgg"]

    timestamp_list = datetime.now().strftime("%Y-%m-%d_%H-%M-%S").split("_")

    hyperparameters = {
        "test_name": test_names[2],
        "model_name": model_names[0],
        "num_folds": 3,
        "num_epochs": 2,
        "batch_size": 32,
        "optimizer_name": "adam",
        "transform_name": "trivial",
        "timestamp_list": timestamp_list,

    }

    # Save hyperparameters in config/timestamp.txt file
    config_dir = config_dir / Path(hyperparameters["timestamp_list"][0])
    config_dir.mkdir(exist_ok=True)
    # Write arguments on config.txt file
    with open(f"{config_dir}/{hyperparameters['timestamp_list'][1]}.txt", "w") as f:
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Setup Device ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple GPU
    else:
        device = torch.device("cpu")  # CPU (if others unavailable)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Pick Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    model_obj = GetModel(
        model_name=hyperparameters["model_name"],
        device=device,
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data Augmentation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    transform_obj = GetTransforms(transform_name=hyperparameters["transform_name"])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Run Test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    print(f"[INFO] Running {hyperparameters['test_name']} test.")

    if hyperparameters["test_name"] == "test_solvers-simple":
        test_solver(
            model_obj=model_obj,
            device=device,
            num_epochs=hyperparameters["num_epochs"],
            batch_size=hyperparameters["batch_size"],
            optimizer_name=hyperparameters["optimizer_name"],
            train_dir=train_dir,
            test_dir=test_dir,
            transform_obj=transform_obj,
            timestamp_list=hyperparameters["timestamp_list"]
        )
    elif hyperparameters["test_name"] == "test_solvers-kfold":
        test_kfold_solver(
            model_obj=model_obj,
            device=device,
            num_folds=hyperparameters["num_folds"],
            num_epochs=hyperparameters["num_epochs"],
            batch_size=hyperparameters["batch_size"],
            optimizer_name=hyperparameters["optimizer_name"],
            train_dir=train_dir,
            test_dir=test_dir,
            transform_obj=transform_obj,
            timestamp_list=hyperparameters["timestamp_list"]
        )
    # elif TEST_NAME == "evaluate":
    #     experiment_name = experiment_names[TEST_NAME]

    #     metrics, infos, experiment_done = test_model(
    #         model_obj=model_obj,
    #         test_dir=results_dir,
    #         num_fold=NUM_FOLDS,
    #         num_epochs=NUM_EPOCHS,
    #         batch_size=BATCH_SIZE,
    #         models_dir=models_dir,
    #         num_workers=NUM_WORKERS if NUM_WORKERS is not None else 1,
    #         test_model_path=test_model_path,
    #         transform_obj=transform_obj,
    #     )

    #     save_results(
    #         root_dir=root_dir,
    #         models_name=model_obj.model_name,
    #         experiment_name=experiment_name,
    #         transform_name=transform_obj.transform_name,
    #         extra=infos,
    #         metrics=metrics
    #     )
    elif hyperparameters["test_name"] == "test_data":
        test_transform(
            data_dir=data_dir,
            transform_obj=transform_obj
        )
