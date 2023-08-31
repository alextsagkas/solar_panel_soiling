import torch
import torch.backends.mps
from pathlib import Path
import os

from packages.models.tiny_vgg import TinyVGG
from packages.utils.transforms import test_data_transform
from packages.utils.load_data import get_dataloader
from packages.utils.training import inference

if __name__ == "__main__":

    # Setup device-agnostic code
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple GPU
    else:
        device = torch.device("cpu")  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

    # Paths
    root_dir = Path("/Users/alextsagkas/Document/Office/solar_panels")
    test_dir = root_dir / "data" / "results"

    # Instantiate the model
    NUM_FOLD = 0
    NUM_EPOCHS = 3
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001

    EXTRA = f"{NUM_FOLD}_f_{NUM_EPOCHS}_e_{BATCH_SIZE}_bs_{HIDDEN_UNITS}_hu_{LEARNING_RATE}_lr"
    MODEL_NAME = "tiny_vgg" + "-" + EXTRA + ".pth"

    model = TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=2
    ).to(device)
    model.load_state_dict(torch.load(f=str(root_dir / "models" / MODEL_NAME)))

    # Load data
    BATCH_SIZE = 1
    NUM_WORKERS = os.cpu_count()

    test_data_transform = test_data_transform

    test_dataloader, class_names = get_dataloader(
        dir=str(test_dir),
        data_transform=test_data_transform,
        BATCH_SIZE=BATCH_SIZE,
        NUM_WORKERS=NUM_WORKERS if NUM_WORKERS else 1,
        shuffle=False
    )

    # Inference
    save_folder = root_dir / "debug" / "test_model"

    test_acc = inference(
        model=model,
        test_dataloader=test_dataloader,
        class_names=class_names,
        save_folder=save_folder,
        model_name="tiny_vgg",
        extra=EXTRA,
        device=device
    )

    print(f"Test accuracy: {test_acc:.4f}")
