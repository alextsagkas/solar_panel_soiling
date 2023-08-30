import os
import torch
import torch.backends.mps
import argparse
from pathlib import Path
from torchvision import datasets

from packages.utils.tensorboard import create_writer
from packages.utils.transforms import train_data_transform, test_data_transform
from packages.utils.training import k_fold_cross_validation

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get hyper-parameters")

    # Get an arg for num_epochs
    parser.add_argument("--num_folds",
                        default=5,
                        type=int,
                        help="the number of fold to spare training data into")

    parser.add_argument("--num_epochs",
                        default=5,
                        type=int,
                        help="the number of epochs to train for")

    # Get an arg for batch_size
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="number of samples per batch")

    # Get an arg for hidden_units
    parser.add_argument("--hidden_units",
                        default=10,
                        type=int,
                        help="number of hidden units in hidden layers")

    # Get an arg for learning_rate
    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float,
                        help="learning rate to use for model")

    # Get our arguments from the parser
    args = parser.parse_args()
    # Write them on config.txt file
    with open("config.txt", "w") as f:
        f.write(args.__str__())

    # Setup hyper-parameters
    NUM_FOLDS = args.num_folds
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    HIDDEN_UNITS = args.hidden_units
    LEARNING_RATE = args.learning_rate

    print(
        f"[INFO] Training a model for {NUM_FOLDS} folds, {NUM_EPOCHS} epochs each, with batch size {BATCH_SIZE} using {HIDDEN_UNITS} hidden units and a learning rate of {LEARNING_RATE}")

    # Setup directories
    root_dir = Path("/Users/alextsagkas/Document/Office/solar_panels")

    train_dir = root_dir / "data" / "train"
    test_dir = root_dir / "data" / "test"
    models_path = root_dir / "models"

    print(f"[INFO] Training data folder: {train_dir}")
    print(f"[INFO] Testing data folder: {test_dir}")

    # Setup device-agnostic code
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple GPU
    else:
        device = torch.device("cpu")  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

    print(f"[INFO] Using {device} device")

    # Instantiate the writer
    EXTRA = f"{NUM_FOLDS}_f_{NUM_EPOCHS}_e_{BATCH_SIZE}_bs_{HIDDEN_UNITS}_hu_{LEARNING_RATE}_lr"

    writer = create_writer(
        experiment_name="test_kfold_tiny_vgg",
        model_name="tiny_vgg",
        extra=EXTRA
    )

    # Set Seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Setup loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()

    # Custom Dataset
    train_data_transform = train_data_transform
    test_data_transform = train_data_transform

    train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=train_data_transform,
        target_transform=None
    )

    test_dataset = datasets.ImageFolder(
        root=str(test_dir),
        transform=test_data_transform,
        target_transform=None
    )

    k_fold_cross_validation(
        model_name="tiny_vgg",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        loss_fn=loss_fn,
        hidden_units=HIDDEN_UNITS,
        device=device,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        root_dir=root_dir,
        optimizer_name="adam",
        num_folds=NUM_FOLDS,
        save_models=True,
        writer=writer
    )
