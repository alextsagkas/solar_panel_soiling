from typing import Dict, Union

import torch
from typing_extensions import Self

from packages.models.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2
from packages.models.resnet import ResNet18, ResNet34, ResNet50
from packages.models.tiny_vgg import TinyVGG, TinyVGGBatchnorm
from packages.utils.configuration import checkpoint_dir
from packages.utils.storage import load_hyperparameters


class GetModel:
    """Class that returns a model based on the model_name parameter.

    Attributes:
        model_name (str): String that identifies the model to be used.
        config (Union[Dict[str, int], None]): Dictionary with the configuration of the model.
        input_shape (int): Number of channels of the input images.
        output_shape (int): Number of classes of the output.
        load_checkpoint (bool): If True, the model is loaded from the checkpoint path, that is 
            specified in the load_config parameter.
        load_config (Union[Dict, None]): Dictionary with the configuration of the loaded model.
        mode_save_path (str): Path to the model parameters checkpoint to be loaded.


    Methods:
        _tiny_vgg: Returns the TinyVGG model.
        _tiny_vgg_batchnorm: Returns the TinyVGGBatchnorm model.
        _resnet18: Returns the ResNet18 model.
        _resnet34: Returns the ResNet34 model.
        _resnet50: Returns the ResNet50 model.
        _efficientnet_b0: Returns the EfficientNetB0 model.
        _efficientnet_b1: Returns the EfficientNetB1 model.
        _efficientnet_b2: Returns the EfficientNetB2 model.
        _load_model: Uses the load_config parameter to get the model name and the configuration
            used while initially training it. It also computes the path of the model parameters
            checkpoint to be loaded.
        get_model: Returns the model based on the model_name parameter.
    """

    def __init__(
        self: Self,
        model_name: str = "",
        config: Union[Dict[str, Union[int, float]], None] = None,
        load_checkpoint: bool = False,
        load_config: Union[Dict, None] = None,
    ) -> None:
        """Initializes the GetModel class.

        Args:
            self (Self): GetModel instance.
            model_name (str, optional): String that identifies the model to be used. Defaults to "".
            config (Union[Dict[str, Union[int, float]], None], optional): Dictionary with the   
                configuration of the model. Defaults to None.
            load_checkpoint (bool, optional): If True, the model is loaded from the checkpoint
                path, that is specified in the load_config parameter. Defaults to False.
            load_config (Union[Dict, None], optional): Dictionary with the configuration of the
                loaded model. An example of a load_config is the following:
                    load_config = {
                        "checkpoint_timestamp_list": ["2021-08-01", "2021-08-01_12-00-00"],
                        "load_epoch": 1,
                    }
                Defaults to None. 
        """
        self.model_name = model_name
        self.config = config
        self.input_shape = 3  # 3 channels (RGB)
        self.output_shape = 2  # 2 classes (clean and soiled)

        self.load_checkpoint = load_checkpoint
        self.load_config = load_config
        self.model_save_path = ""

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

    def _resnet34(
        self: Self,
    ) -> torch.nn.Module:
        """Returns the ResNet34 model with pretrained the inner layers. Only the last 
        (classification) layer is trainable and outputs 2 classes.

        Args:
            self (Self): GetModel instance.

        Returns:
            torch.nn.Module: The ResNet34 model.
        """

        print(
            "[INFO] Using ResNet34 model with: "
            "pre-trained weights in all layers, but the classifier."
        )

        return ResNet34()

    def _resnet50(
        self: Self,
    ) -> torch.nn.Module:
        """Returns the ResNet50 model with pretrained the inner layers. Only the last 
        (classification) layer is trainable and outputs 2 classes.

        Args:
            self (Self): GetModel instance.

        Returns:
            torch.nn.Module: The ResNet50 model.
        """

        print(
            "[INFO] Using ResNet50 model with: "
            "pre-trained weights in all layers, but the classifier."
        )

        return ResNet50()

    def _efficientnet_b0(
        self: Self,
    ) -> torch.nn.Module:
        """Returns the EfficientNetB0 model with pretrained the inner layers. Only the last 
        (classification) layer is trainable and outputs 2 classes.

        Args:
            self (Self): GetModel instance.

        Returns:
            torch.nn.Module: The EfficientNetB0 model.
        """

        print(
            "[INFO] Using EfficientNetB0 model with: "
            "pre-trained weights in all layers, but the classifier."
        )

        return EfficientNetB0()

    def _efficientnet_b1(
        self: Self,
    ) -> torch.nn.Module:
        """Returns the EfficientNetB1 model with pretrained the inner layers. Only the last 
        (classification) layer is trainable and outputs 2 classes.

        Args:
            self (Self): GetModel instance.

        Returns:
            torch.nn.Module: The EfficientNetB1 model.
        """

        print(
            "[INFO] Using EfficientNetB1 model with: "
            "pre-trained weights in all layers, but the classifier."
        )

        return EfficientNetB1()

    def _efficientnet_b2(
        self: Self,
    ) -> torch.nn.Module:
        """Returns the EfficientNetB2 model with pretrained the inner layers. Only the last 
        (classification) layer is trainable and outputs 2 classes.

        Args:
            self (Self): GetModel instance.

        Returns:
            torch.nn.Module: The EfficientNetB2 model.
        """

        print(
            "[INFO] Using EfficientNetB2 model with: "
            "pre-trained weights in all layers, but the classifier."
        )

        return EfficientNetB2()

    def _load_model(
        self: Self,
    ) -> None:
        """Uses the load_config parameter to get the model name and the configuration used while 
        initially training it. It also computes the path of the model parameters checkpoint to be 
        loaded. 

        The following parameters are saved in class' attributes:
            model_name: String that identifies the model to be used.
            config: Dictionary with the configuration of the model.
            model_save_path: Path to the model parameters checkpoint to be loaded.

        Args:
            self (Self): Instance of GetModel.

        Raises:
            ValueError: If load_config is None.
        """
        if self.load_config == None:
            raise ValueError("load_config must be provided.")
        checkpoint_timestamp_list = self.load_config["checkpoint_timestamp_list"]
        load_epoch = self.load_config["load_epoch"]

        # Load test hyperparameters
        test_hyperparameters = load_hyperparameters(
            test_timestamp_list=checkpoint_timestamp_list,
        )

        # Extract hyperparameters from initial model training
        self.model_name = test_hyperparameters["model_name"]
        self.config: Union[Dict, None] = test_hyperparameters.get("model_config", None)  # type: ignore

        # Load model
        self.model_save_path = (
            checkpoint_dir /
            checkpoint_timestamp_list[0] /
            f"{checkpoint_timestamp_list[1]}_epoch_{load_epoch}.pth")  # type: ignore

    def get_model(
        self: Self,
    ) -> torch.nn.Module:
        """Returns the model based on the model_name parameter. If load_checkpoint is True, the 
        model is loaded from the checkpoint path.

        Args:
            self (Self): GetModel instance.

        Returns:
            torch.nn.Module: The model to be used.

        Raises:
            ValueError: If the model_name is not supported (there is no method that implements it).
        """
        if self.load_checkpoint:
            self._load_model()

        model_method_name = f"_{self.model_name}"
        model_method = getattr(self, model_method_name, None)

        if model_method is not None and callable(model_method):

            model = model_method()

            if self.load_checkpoint:
                model.load_state_dict(torch.load(f=self.model_save_path))

                print(f"[INFO] Model loaded from {self.model_save_path}.")

            return model

        else:
            raise ValueError(f"Model {self.model_name} is not supported.")
