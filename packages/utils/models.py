from typing import Dict, Union

import torch
from typing_extensions import Self

from packages.models.tiny_vgg import TinyVGG


class GetModel:
    """Class that returns a model based on the model_name parameter.

    Args:
        model_name (str): String that identifies the model to be used.
        device (torch.device): Device to be used to load the model.
        config (Union[Dict[str, int], None], optional): Dictionary with the configuration of the
            model. Defaults to None.

    Attributes:
        model_name (str): String that identifies the model to be used.
        device (torch.device): Device to be used to load the model.
        config (Union[Dict[str, int], None]): Dictionary with the configuration of the model.

    Methods:
        _tiny_vgg: Returns the TinyVGG model.
        get_model: Returns the model based on the model_name parameter.
    """

    def __init__(
        self: Self,
        model_name: str,
        config: Union[Dict[str, int], None] = None,
    ) -> None:
        """Initializes the GetModel class.

        Args:
            self (Self): GetModel instance.
            model_name (str): String that identifies the model to be used.
            config (Union[Dict[str, int], None], optional): Dictionary with the configuration of
                the model. Defaults to None.
        """
        self.model_name = model_name
        self.config = config
        self.input_shape = 3  # 3 channels (RGB)
        self.output_shape = 2  # 2 classes (clean and soiled)

    def _tiny_vgg(
        self: Self,
    ) -> torch.nn.Module:
        """Returns the TinyVGG model.

        Args:
            self (Self): GetModel instance.

        Returns:
            torch.nn.Module: TinyVGG model.
        """
        if self.config is None:
            self.config = {}
        self.config.setdefault("hidden_units", 32)

        print(
            "[INFO] Using TinyVGG model with "
            f"{self.config['hidden_units']} hidden units."
        )

        return TinyVGG(
            input_shape=self.input_shape,
            hidden_units=self.config["hidden_units"],
            output_shape=self.output_shape,
        )

    def get_model(
        self: Self,
    ) -> torch.nn.Module:
        """Returns the model based on the model_name parameter.

        Args:
            self (Self): GetModel instance.

        Returns:
            torch.nn.Module: The model to be used.
        """
        model_method_name = f"_{self.model_name}"
        model_method = getattr(self, model_method_name, None)

        if model_method is not None and callable(model_method):
            return model_method()

        else:
            raise ValueError(f"Model {self.model_name} is not supported.")
