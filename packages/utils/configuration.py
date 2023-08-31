import torch.nn
from torch import optim

from packages.models.tiny_vgg import TinyVGG


def _get_model(
    model_name: str,
    hidden_units: int
) -> torch.nn.Module:
    """Returns a model based on the model_name and hidden_units parameters. The list of available models is: "tiny_vgg".

    Args:
        model_name (str): String that identifies the model to be used.
        hidden_units (int): Number of hidden units to be used in the model.

    Raises:
        ValueError: If the model_name is not one of "tiny_vgg".

    Returns:
        torch.nn.Module: The model to be used.
    """
    if model_name == "tiny_vgg":
        model = TinyVGG(
            input_shape=3,
            hidden_units=hidden_units,
            output_shape=2
        )
    else:
        raise ValueError(
            f"Model name {model_name} is not supported. "
            "Please choose between 'tiny_vgg'."
        )

    return model


def _get_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    learning_rate: float = 1e-3,
) -> torch.optim.Optimizer:
    """Returns an optimizer based on the optimizer_name and learning_rate parameters. The list of available optimizers is: "adam", "sgd".

    Args:
        model (torch.nn.Module): Model to be optimized (the parameters of it are needed).
        optimizer_name (str): String that identifies the optimizer to be used.
        learning_rate (float, optional): Learning rate to be used in updates. Defaults to 1e-3.

    Raises:
        ValueError: If the optimizer_name is not one of "adam", "sgd".

    Returns:
        torch.optim.Optimizer: The optimizer to be used.
    """
    if optimizer_name == "adam":
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=learning_rate
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=learning_rate
        )
    else:
        raise ValueError(
            f"Optimizer name {optimizer_name} is not supported. "
            "Please choose between 'adam' and 'sgd'."
        )

    return optimizer
