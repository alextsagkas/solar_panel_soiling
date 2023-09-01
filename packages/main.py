import argparse
import os
from pathlib import Path

import torch
import torch.backends.mps

from packages.tests.test_cross_validation import test_cross_validation
from packages.tests.test_model import test_model
from packages.tests.test_train import test_train
from packages.utils.storage import save_results

if __name__ == "__main__":

    # * Setup Parser
    parser = argparse.ArgumentParser(description="Get hyper-parameters")

    # Get an arg for the test called
    parser.add_argument("--tn",
                        default="train",
                        type=str,
                        help="the test to run")

    # Get an arg for model name
    parser.add_argument("--mn",
                        default="tiny_vgg",
                        type=str,
                        help="the name of the model")

    # Get an arg for number of folds
    parser.add_argument("--f",
                        default=-1,
                        type=int,
                        help="the number of folds to use in k-fold cross validation")

    # Get an arg for number of epochs
    parser.add_argument("--e",
                        default=5,
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

    args = parser.parse_args()

    # Write arguments on config.txt file
    with open("config.txt", "w") as f:
        f.write(args.__str__())

    # * Setup hyper-parameters
    test_names = ["train", "model", "data", "cross_validation"]
    if args.tn not in test_names:
        raise ValueError(f"Test name must be one of {test_names}")
    else:
        TEST_NAME = args.tn

    model_names = ["tiny_vgg"]
    if args.mn not in model_names:
        raise ValueError(f"Model name must be one of {model_names}")
    else:
        MODEL_NAME = args.mn

    if args.f < 2 and args.tn in ["cross_validation"]:
        raise ValueError("Number of folds must be greater than 1")
    elif args.f == -1 or args.tn in ["train"]:
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

    if args.hu < 1:
        raise ValueError("Number of hidden units must be greater than 0")
    else:
        HIDDEN_UNITS = args.hu

    optimizer_names = ["adam", "sgd"]
    if args.on not in optimizer_names:
        raise ValueError(f"Optimizer name must be one of {optimizer_names}")
    else:
        OPTIMIZER_NAME = args.on

    if args.lr < 0 or args.lr > 1:
        raise ValueError("Learning rate must be between 0 and 1")
    else:
        LEARNING_RATE = args.lr

    print(
        f"[INFO] Using the {MODEL_NAME} model{folds_text} for {NUM_EPOCHS} epochs, with batch size {BATCH_SIZE}, using {HIDDEN_UNITS} hidden units and a learning rate of {LEARNING_RATE}"
    )

    # * Setup directories
    root_dir = Path("/Users/alextsagkas/Document/Office/solar_panels")

    # Data
    train_dir = root_dir / "data" / "train"
    test_dir = root_dir / "data" / "test"
    results_dir = root_dir / "data" / "results"

    print(f"[INFO] Training data file: {train_dir}")
    print(f"[INFO] Testing data file: {test_dir}")
    print(f"[INFO] Results data file: {results_dir}")

    # Models
    models_path = root_dir / "models"
    print(f"[INFO] Models path: {models_path}")

    # Test Model Path
    test_model_path = root_dir / "debug" / "test_model"

    # * Setup device-agnostic code
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple GPU
    else:
        device = torch.device("cpu")  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

    print(f"[INFO] Using {device} device")

    # Setup data loaders
    if os.cpu_count() is None:
        raise ValueError("Number of workers must be greater than 0")
    else:
        NUM_WORKERS = os.cpu_count()

    print(f"[INFO] Using {NUM_WORKERS} workers to load data")

    # * Run tests

    print(f"[INFO] Running {TEST_NAME} test")

    if TEST_NAME == "train":
        metrics, infos, experiment_name = test_train(
            model_name=MODEL_NAME,
            train_dir=train_dir,
            test_dir=test_dir,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS if NUM_WORKERS is not None else 1,
            num_epochs=NUM_EPOCHS,
            hidden_units=HIDDEN_UNITS,
            optimizer_name=OPTIMIZER_NAME,
            learning_rate=LEARNING_RATE,
            device=device,
            models_path=models_path
        )

        save_results(
            root_dir=root_dir,
            models_name=MODEL_NAME,
            experiment_name=experiment_name,
            extra=infos,
            metrics=metrics
        )
    elif TEST_NAME == "cross_validation" and NUM_FOLDS != -1:
        metrics, infos, experiment_name = test_cross_validation(
            root_dir=root_dir,
            models_path=models_path,
            train_dir=train_dir,
            test_dir=test_dir,
            device=device,
            num_folds=NUM_FOLDS,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            hidden_units=HIDDEN_UNITS,
            learning_rate=LEARNING_RATE,
            model_name=MODEL_NAME,
            optimizer_name=OPTIMIZER_NAME,
            save_models=True
        )

        save_results(
            root_dir=root_dir,
            models_name=MODEL_NAME,
            experiment_name=experiment_name,
            extra=infos,
            metrics=metrics
        )
    elif TEST_NAME == "model":
        metrics, infos, experiment_name = test_model(
            model_name=MODEL_NAME,
            test_dir=results_dir,
            num_fold=NUM_FOLDS,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            hidden_units=HIDDEN_UNITS,
            learning_rate=LEARNING_RATE,
            device=device,
            models_path=models_path,
            num_workers=NUM_WORKERS if NUM_WORKERS is not None else 1,
            test_model_path=test_model_path
        )

        save_results(
            root_dir=root_dir,
            models_name=MODEL_NAME,
            experiment_name=experiment_name,
            extra=infos,
            metrics=metrics
        )
