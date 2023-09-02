from typing import Dict, Iterator, Union

import torch.optim
from torch import optim
from typing_extensions import Self

from packages.models.tiny_vgg import TinyVGG


class GetOptimizer:
    """Class that returns an optimizer based on the optimizer_name parameter. The list of available
    optimizers is: ["adam", "sgd"].

    Args:
        model (torch.nn.Module): The model to be optimized.
        optimizer_name (str): String that identifies the optimizer to be used.
        config (Union[Dict[str, float], None], optional): Dictionary with the configuration of the
            optimizer. Defaults to None.

    Attributes:
        optimizer_name (str): String that identifies the optimizer to be used.
        params (Iterator[torch.nn.Parameter]): List of parameters of the model to be optimized.
        config (Union[Dict[str, float], None]): Dictionary with the configuration of the optimizer. 

    Methods:
        _sgd: Returns a SGD optimizer.
        _adam: Returns an Adam optimizer.
        get_optimizer: Returns the optimizer based on the optimizer_name parameter.
    """

    def __init__(
        self: Self,
        params: Iterator[torch.nn.Parameter],
        optimizer_name: str,
        config: Union[Dict[str, float], None] = None,
    ) -> None:
        """Initializes the GetOptimizer class.

        Args:
            self (Self): GetOptimizer instance.
            params (List[torch.nn.Parameter]): List of parameters of the model to be optimized 
            optimizer_name (str): String that identifies the optimizer to be used (should match
                with a method implemented below, e.g. "sgd", "adam").
            config (Union[Dict[str, float], None], optional): Dictionary with the configuration 
                of the optimizer. Defaults to None.
        """
        self.optimizer_name = optimizer_name
        self.params = params
        self.config = config

    def _sgd(
        self: Self,
    ) -> torch.optim.Optimizer:
        """Returns a SGD optimizer.

        Returns:
            torch.optim.Optimizer: SGD optimizer.
        """
        if self.config is None:
            self.config = {}
        self.config.setdefault("learning_rate", 1e-2)

        print(f"[INFO] Using SGD optimizer with lr={self.config['learning_rate']}")

        return optim.SGD(
            params=self.params,
            lr=self.config["learning_rate"],
        )

    def _adam(
        self: Self,
    ) -> torch.optim.Optimizer:
        """Returns an Adam optimizer.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        if self.config is None:
            self.config = {}
        self.config.setdefault("learning_rate", 1e-3)
        self.config.setdefault("beta1", 0.9)
        self.config.setdefault("beta2", 0.999)
        self.config.setdefault("epsilon", 1e-8)

        print(
            f"[INFO] Using Adam optimizer with "
            f"lr={self.config['learning_rate']}, "
            f"beta1={self.config['beta1']}, "
            f"beta2={self.config['beta2']}, "
            f"epsilon={self.config['epsilon']}."
        )

        return optim.Adam(
            params=self.params,
            lr=self.config["learning_rate"],
            betas=(self.config["beta1"], self.config["beta2"]),
            eps=self.config["epsilon"],
        )

    def get_optimizer(
        self: Self,
    ) -> torch.optim.Optimizer:
        """Returns the optimizer based on the optimizer_name attribute.

        Returns:
            torch.optim.Optimizer: Optimizer instance.

        Raises:
            ValueError: When the optimizer_name does not correspond to any optimizer method.
        """
        optimizer_method_name = f"_{self.optimizer_name}"
        optimizer_method = getattr(self, optimizer_method_name, None)

        if optimizer_method is not None and callable(optimizer_method):
            return optimizer_method()
        else:
            raise ValueError(f"Optimizer '{self.optimizer_name}' is not supported")


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

        # Delete from memory models that are not returned
        for key, _ in models_dict.items():
            if key != self.model_name:
                del models_dict[key]

        return models_dict[self.model_name].to(self.device)
