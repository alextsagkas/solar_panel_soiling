import torch
import torchvision
from torch import nn
from typing_extensions import Self


class ShuffleNetV2X05(nn.Module):
    """Class that returns a ShuffleNet_V2_X0_5 model with the last layer changed to output 2 
    classes.

    Attributes:
        weights (str): Downloaded weights for the ShuffleNet_V2_X0_5 model from IMAGENET1K_V1.
        model (torch.nn.Module): ShuffleNet_V2_X0_5 model with the pretrained weights on all layers.

    Methods:
        forward: Passes the input tensor x through the model.
    """

    def __init__(
        self: Self,
    ) -> None:
        """Download the IMAGENET1K_V1 weights for ShuffleNet_V2_X0_5 and initializes the model with 
        the pre-trained weights on all layers. Afterwards, freeze all layers except the last
        one and change it to output 2 classes.
        """
        super().__init__()

        self.weights = torchvision.models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
        # Load the model with the pretrained weights
        self.model = torchvision.models.shufflenet_v2_x0_5(
            weights=self.weights,
            progress=True,
        )
        # Freeze the all layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Change the classifier layer to output 2 classes
        self.model.fc = nn.Linear(
            in_features=1024,
            out_features=2,
            bias=True
        )

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> torch.nn.Module:
        """Passes the input tensor x through the model.

        Args:
            self (Self): Instance of the class.
            x (torch.Tensor): Input tensor to pass it through the model.

        Returns:
            torch.nn.Module: Output of the model.
        """

        return self.model(x)


class ShuffleNetV2X10(nn.Module):
    """Class that returns a ShuffleNet_V2_X1_0 model with the last layer changed to output 2 
    classes.

    Attributes:
        weights (str): Downloaded weights for the ShuffleNet_V2_X1_0 model from IMAGENET1K_V1.
        model (torch.nn.Module): ShuffleNet_V2_X1_0 model with the pretrained weights on all layers.

    Methods:
        forward: Passes the input tensor x through the model.
    """

    def __init__(
        self: Self,
    ) -> None:
        """Download the IMAGENET1K_V1 weights for ShuffleNet_V2_X1_0 and initializes the model with 
        the pre-trained weights on all layers. Afterwards, freeze all layers except the last
        one and change it to output 2 classes.
        """
        super().__init__()

        self.weights = torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
        # Load the model with the pretrained weights
        self.model = torchvision.models.shufflenet_v2_x1_0(
            weights=self.weights,
            progress=True,
        )
        # Freeze the all layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Change the classifier layer to output 2 classes
        self.model.fc = nn.Linear(
            in_features=1024,
            out_features=2,
            bias=True
        )

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> torch.nn.Module:
        """Passes the input tensor x through the model.

        Args:
            self (Self): Instance of the class.
            x (torch.Tensor): Input tensor to pass it through the model.

        Returns:
            torch.nn.Module: Output of the model.
        """

        return self.model(x)


class ShuffleNetV2X15(nn.Module):
    """Class that returns a ShuffleNet_V2_X1_5 model with the last layer changed to output 2 
    classes.

    Attributes:
        weights (str): Downloaded weights for the ShuffleNet_V2_X1_5 model from IMAGENET1K_V1.
        model (torch.nn.Module): ShuffleNet_V2_X1_5 model with the pretrained weights on all layers.

    Methods:
        forward: Passes the input tensor x through the model.
    """

    def __init__(
        self: Self,
    ) -> None:
        """Download the IMAGENET1K_V1 weights for ShuffleNet_V2_X1_5 and initializes the model with 
        the pre-trained weights on all layers. Afterwards, freeze all layers except the last
        one and change it to output 2 classes.
        """
        super().__init__()

        self.weights = torchvision.models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1
        # Load the model with the pretrained weights
        self.model = torchvision.models.shufflenet_v2_x1_5(
            weights=self.weights,
            progress=True,
        )
        # Freeze the all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # TODO: Make this work with the API -- Configure from main
        for param in self.model.conv5.parameters():
            param.requires_grad = True  # type: ignore

        # Change the classifier layer to output 2 classes
        self.model.fc = nn.Linear(
            in_features=1024,
            out_features=2,
            bias=True
        )

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> torch.nn.Module:
        """Passes the input tensor x through the model.

        Args:
            self (Self): Instance of the class.
            x (torch.Tensor): Input tensor to pass it through the model.

        Returns:
            torch.nn.Module: Output of the model.
        """

        return self.model(x)


class ShuffleNetV2X20(nn.Module):
    """Class that returns a ShuffleNet_V2_X2_0 model with the last layer changed to output 2 
    classes.

    Attributes:
        weights (str): Downloaded weights for the ShuffleNet_V2_X2_0 model from IMAGENET1K_V1.
        model (torch.nn.Module): ShuffleNet_V2_X2_0 model with the pretrained weights on all layers.

    Methods:
        forward: Passes the input tensor x through the model.
    """

    def __init__(
        self: Self,
    ) -> None:
        """Download the IMAGENET1K_V1 weights for ShuffleNet_V2_X2_0 and initializes the model with 
        the pre-trained weights on all layers. Afterwards, freeze all layers except the last
        one and change it to output 2 classes.
        """
        super().__init__()

        self.weights = torchvision.models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1
        # Load the model with the pretrained weights
        self.model = torchvision.models.shufflenet_v2_x2_0(
            weights=self.weights,
            progress=True,
        )
        # Freeze the all layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Change the classifier layer to output 2 classes
        self.model.fc = nn.Linear(
            in_features=2048,
            out_features=2,
            bias=True
        )

    def forward(
        self: Self,
        x: torch.Tensor
    ) -> torch.nn.Module:
        """Passes the input tensor x through the model.

        Args:
            self (Self): Instance of the class.
            x (torch.Tensor): Input tensor to pass it through the model.

        Returns:
            torch.nn.Module: Output of the model.
        """

        return self.model(x)
