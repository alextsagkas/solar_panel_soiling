import torch.nn
import torch.optim
from torch import optim
from typing_extensions import Self

from packages.models.tiny_vgg import TinyVGG


class GetModel:
    """Class that returns a model based on the model_name parameter. The list of available models 
    is: ["tiny_vgg"].

    Args:
        model_name (str): String that identifies the model to be used.
        hidden_units (int): Number of hidden units in the hidden layers.
        device (torch.device): Device to be used to load the model.

    Attributes:
        model_name_list (List[str]): List of available models.
        model_name (str): String that identifies the model to be used.
        hidden_units (int): Number of hidden units in the hidden layers.
        device (torch.device): Device to be used to load the model.
        input_shape (int): Number of color channels of the input images.
        output_shape (int): Number of classes of the output.

    Methods:
        get_model: Returns the model based on the model_name parameter.
    """

    def __init__(
        self: Self,
        model_name: str,
        hidden_units: int,
        device: torch.device,
    ) -> None:
        """Initializes the GetModel class.

        Args:
            self (Self): GetModel instance.
            model_name (str): String that identifies the model to be used.
            hidden_units (int): Number of hidden units in the hidden layers.
            device (torch.device): Device to be used to load the model.

        Raises:
            ValueError: When the model_name is not one of the available models.
            ValueError: When the number of hidden units is less than 0.
            ValueError: When the device is not one of the available devices ("cpu", "cuda", "mps").
        """
        self.model_name_list = ["tiny_vgg"]
        if model_name not in self.model_name_list:
            raise ValueError(f"model_name must be one of {self.model_name_list}")
        else:
            self.model_name = model_name

        if hidden_units < 0:
            raise ValueError("Number of hidden units must be greater than 0")
        else:
            self.hidden_units = hidden_units

        device_list = [
            torch.device("cpu"),
            torch.device("cuda"),
            torch.device("mps"),
        ]
        if device not in device_list:
            raise ValueError(f"device must be one of {device_list}")
        else:
            self.device = device

        print(
            f"[INFO] Using {self.model_name} model, "
            f"with {hidden_units} hidden units, "
            f"on {self.device} device."
        )

        self.input_shape = 3  # 3 channels (RGB)
        self.output_shape = 2  # 2 classes (clean and soiled)

    def get_model(
        self: Self,
    ) -> torch.nn.Module:
        """Returns the model based on the model_name parameter.

        Args:
            self (Self): GetModel instance.

        Returns:
            torch.nn.Module: The model to be used.
        """
        models_dict = {
            "tiny_vgg": TinyVGG(
                input_shape=self.input_shape,
                hidden_units=self.hidden_units,
                output_shape=self.output_shape,
            )
        }

        return models_dict[self.model_name].to(self.device)


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


class GetOptimizer:
    # TODO: add docstring
    def __init__(
        # TODO: add docstring
        self: Self,
        model: torch.nn.Module,
        optimizer_name: str,
        learning_rate: float = 1e-3,
    ) -> None:
        self.optimizer_name_list = ["adam", "sgd"]
        if optimizer_name not in self.optimizer_name_list:
            raise ValueError(f"optimizer_name must be one of {self.optimizer_name_list}")
        else:
            self.optimizer_name = optimizer_name

        self.model_params = model.parameters()

        if learning_rate < 0 or learning_rate > 1:
            raise ValueError("Learning rate must be between 0 and 1")
        else:
            self.learning_rate = learning_rate

    def get_optimizer(
        self: Self,
    ) -> torch.optim.Optimizer:
        # TODO: add docstring
        optimizer_dict = {
            "adam": optim.Adam(
                params=self.model_params,
                lr=self.learning_rate
            ),
            "sgd": optim.SGD(
                params=self.model_params,
                lr=self.learning_rate
            )
        }

        return optimizer_dict[self.optimizer_name]
