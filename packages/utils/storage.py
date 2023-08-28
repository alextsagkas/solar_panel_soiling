import torch
from pathlib import PosixPath


def save_model(
    model: torch.nn.Module,
    MODELS_PATH: PosixPath,
    MODEL_NAME: str,

) -> None:
    """Saves the state dict of model in MODELS_PATH/MODEL_NAME.

    Args:
        model (torch.nn.Module): Model to save.
        MODELS_PATH (PosixPath): Path the models are saved to.
        MODEL_NAME (str): Name of the model to save.
    """
    MODELS_PATH.mkdir(exist_ok=True)
    MODEL_SAVE_PATH = MODELS_PATH / MODEL_NAME

    print(f"Saving model to: {MODEL_SAVE_PATH}")

    torch.save(
        obj=model.state_dict(),
        f=MODEL_SAVE_PATH
    )
