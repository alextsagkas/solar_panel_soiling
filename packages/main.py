import os
import torch
import argparse
from pathlib import Path

from packages.models.tiny_vgg import TinyVGG
from packages.utils.load_data import get_dataloaders
from packages.utils.storage import save_model
from packages.utils.tensorboard import create_writer
from packages.utils.training import train
from packages.utils.transforms import train_data_transform, test_data_transform

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get hyper-parameters")

    # Get an arg for num_epochs
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
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    HIDDEN_UNITS = args.hidden_units
    LEARNING_RATE = args.learning_rate

    print(
        f"[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} using {HIDDEN_UNITS} hidden units and a learning rate of {LEARNING_RATE}")

    # Setup directories
    root_dir = Path("/Users/alextsagkas/Document/Office/solar_panels")

    train_dir = root_dir / "data" / "train"
    test_dir = root_dir / "data" / "test"
    models_path = root_dir / "models"

    print(f"[INFO] Training data file: {train_dir}")
    print(f"[INFO] Testing data file: {test_dir}")

    # Setup device-agnostic code
    if torch.cuda.is_available():
        device = "cuda"  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple GPU
    else:
        device = "cpu"  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

    print(f"[INFO] Using {device} device")

    # Setup data loaders
    train_dataloader, test_dataloader = get_dataloaders(
        train_dir=train_dir,
        train_transform=train_data_transform,
        test_dir=test_dir,
        test_transform=test_data_transform,
        BATCH_SIZE=BATCH_SIZE,
        NUM_WORKERS=os.cpu_count()
    )

    # Instantiate the model
    model = TinyVGG(
        input_shape=3,
        hidden_units=32,
        output_shape=2
    ).to(device)

    # Instantiate the writer
    writer = create_writer(
        experiment_name="initial_test_tiny_vgg",
        model_name="tiny_vgg",
        extra=f"{NUM_EPOCHS}_e_{BATCH_SIZE}_bs_{HIDDEN_UNITS}_hu_{LEARNING_RATE}_lr"
    )

    # Set Seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Setup loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device,
        writer=writer
    )

    # Save the model
    save_model(
        model=model,
        MODELS_PATH=models_path,
        MODEL_NAME="tiny_vgg.pth",
    )
