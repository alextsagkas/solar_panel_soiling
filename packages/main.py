import argparse
import os
from pathlib import Path

import torch
import torch.backends.mps

from packages.tests.test_data import test_transform
from packages.tests.test_model import test_model
from packages.tests.test_solvers import test_kfold_solver, test_solver
from packages.utils.models import GetModel
from packages.utils.storage import save_results
from packages.utils.transforms import GetTransforms

if __name__ == "__main__":

    # * Setup Parser
    parser = argparse.ArgumentParser(description="Get hyper-parameters")

    # Get an arg for the test called
    parser.add_argument("--tn",
                        type=str,
                        help="the test to run")

    # Get an arg for model name
    parser.add_argument("--mn",
                        type=str,
                        help="the name of the model")

    # Get an arg for number of folds
    parser.add_argument("--f",
                        default=-1,
                        type=int,
                        help="the number of folds to use in k-fold cross validation")

    # Get an arg for number of epochs
    parser.add_argument("--e",
                        type=int,
                        help="the number of epochs to train for")

    # Get an arg for batch size
    parser.add_argument("--bs",
                        default=32,
                        type=int,
                        help="number of samples per batch")

    # Get an arg for hidden units
    parser.add_argument("--hu",
                        default=10,
                        type=int,
                        help="number of hidden units in hidden layers")

    # Get an arg for optimizer name
    parser.add_argument("--on",
                        default="adam",
                        type=str,
                        help="optimizer to update the model")

    # Get an arg for learning rate
    parser.add_argument("--lr",
                        default=1e-3,
                        type=float,
                        help="learning rate to use for model")

    parser.add_argument("--tr",
                        default="simple",
                        type=str,
                        help="the transform to use for the data")

    args = parser.parse_args()

    # Write arguments on config.txt file
    with open("config.txt", "w") as f:
        f.write(args.__str__())

    # * Setup device-agnostic code
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple GPU
    else:
        device = torch.device("cpu")  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

    print(f"[INFO] Using {device} device")

    # * Setup hyper-parameters

    test_names = ["evaluate", "transform", "solver", "kfold_solver"]
    if args.tn not in test_names:
        raise ValueError(f"Test name must be one of {test_names}")
    else:
        TEST_NAME = args.tn

    model_obj = GetModel(
        model_name=args.mn,
        device=device,
    )

    if args.f < 2 and args.tn in ["kfold"]:
        raise ValueError("Number of folds must be greater than 1")
    elif args.f == -1 or args.tn in ["train", "solver"]:
        folds_text = ""
        NUM_FOLDS = -1
    else:
        NUM_FOLDS = args.f
        folds_text = f" {NUM_FOLDS} folds and"

    if args.e < 1:
        raise ValueError("Number of epochs must be greater than 0")
    else:
        NUM_EPOCHS = args.e

    if (args.bs & (args.bs - 1)) != 0 or args.bs < 1:
        raise ValueError("Batch size must be a positive power of two")
    else:
        BATCH_SIZE = args.bs

    transform_obj = GetTransforms(transform_name=args.tr)

    # * Setup directories
    root_dir = Path("/Users/alextsagkas/Document/Office/solar_panels")

    # Data
    data_dir = root_dir / "data"
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    results_dir = data_dir / "results"

    print(f"[INFO] Training data file: {train_dir}")
    print(f"[INFO] Testing data file: {test_dir}")
    print(f"[INFO] Results data file: {results_dir}")

    # Models
    models_path = root_dir / "models"
    print(f"[INFO] Models path: {models_path}")

    # Test Model Path
    test_model_path = root_dir / "debug" / "test_model"

    # Setup data loaders
    if os.cpu_count() is None:
        raise ValueError("Number of workers must be greater than 0")
    else:
        NUM_WORKERS = os.cpu_count()

    print(f"[INFO] Using {NUM_WORKERS} workers to load data")

    # * Run tests

    experiment_names = {
        "train": "test_train",
        "evaluate": "test_model",
        "transform": "test_data",
        "kfold": "test_kfold"
    }

    print(f"[INFO] Running {TEST_NAME} test")

    if TEST_NAME == "evaluate":
        experiment_name = experiment_names[TEST_NAME]

        metrics, infos, experiment_done = test_model(
            model_obj=model_obj,
            test_dir=results_dir,
            num_fold=NUM_FOLDS,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            models_path=models_path,
            num_workers=NUM_WORKERS if NUM_WORKERS is not None else 1,
            test_model_path=test_model_path,
            transform_obj=transform_obj,
        )

        save_results(
            root_dir=root_dir,
            models_name=model_obj.model_name,
            experiment_name=experiment_name,
            transform_name=transform_obj.transform_name,
            extra=infos,
            metrics=metrics
        )
    elif TEST_NAME == "transform":
        test_transform(
            data_dir=data_dir,
            transform_obj=transform_obj
        )
    elif TEST_NAME == "solver":
        test_solver(
            model_obj=model_obj,
            device=device,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            optimizer_name=args.on,
            train_dir=train_dir,
            test_dir=test_dir,
            transform_obj=transform_obj,
            root_dir=root_dir,
        )
    elif TEST_NAME == "kfold_solver":
        test_kfold_solver(
            model_obj=model_obj,
            device=device,
            num_folds=NUM_FOLDS,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            optimizer_name=args.on,
            train_dir=train_dir,
            test_dir=test_dir,
            transform_obj=transform_obj,
            root_dir=root_dir,
        )
