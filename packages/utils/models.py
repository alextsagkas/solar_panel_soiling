from typing import Dict, Union

import torch
from typing_extensions import Self

from packages.models.resnet import ResNet18
from packages.models.tiny_vgg import TinyVGG, TinyVGGBatchnorm


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
        config: Union[Dict[str, Union[int, float]], None] = None,
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
        if self.config is None:
            self.config = {}
        self.config.setdefault("hidden_units", 32)
        self.config.setdefault("dropout_rate", 0.5)

        print(
            "[INFO] Using TinyVGG model with "
            f"{self.config['hidden_units']} hidden units and "
            f"{self.config['dropout_rate']} dropout rate."
        )

        return TinyVGG(
            dropout_rate=float(self.config["dropout_rate"]),
            input_shape=self.input_shape,
            hidden_units=int(self.config["hidden_units"]),
            output_shape=self.output_shape,
        )

    def _tiny_vgg_batchnorm(
        self: Self,
    ) -> torch.nn.Module:
        if self.config is None:
            self.config = {}
        self.config.setdefault("hidden_units", 32)
        self.config.setdefault("dropout_rate", 0.5)

        print(
            "[INFO] Using TinyVGGBatchnorm model with "
            f"{self.config['hidden_units']} hidden units and "
            f"{self.config['dropout_rate']} dropout rate."
        )

        return TinyVGGBatchnorm(
            dropout_rate=float(self.config["dropout_rate"]),
            input_shape=self.input_shape,
            hidden_units=int(self.config["hidden_units"]),
            output_shape=self.output_shape,
        )

    def _resnet18(
        self: Self,
    ) -> torch.nn.Module:
        """Returns the ResNet18 model with pretrained the inner layers. Only the last 
        (classification) layer is trainable and outputs 2 classes.

        Args:
            self (Self): GetModel instance.

        Returns:
            torch.nn.Module: The ResNet18 model.
        """

        print(
            "[INFO] Using ResNet18 model with: "
            "pre-trained weights in all layers, but the classifier."
        )

        return ResNet18()

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
