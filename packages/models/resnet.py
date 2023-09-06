import torch
import torchvision
from torch import nn
from typing_extensions import Self


class ResNet18(nn.Module):
    """Class that returns a ResNet18 model with the last layer changed to output 2 classes.

    Attributes:
        weights (str): Downloaded weights for the ResNet18 model from IMAGENET1K_V1.
        model (torch.nn.Module): ResNet18 model with the pretrained weights on all layers.

    Methods:
        forward: Passes the input tensor x through the model.
    """

    def __init__(
        self: Self,
    ) -> None:
        """Download the IMAGENET1K_V1 weights for ResNet18 and initializes the model with 
        the pre-trained weights on all layers. Afterwards, freeze all layers except the last
        one and change it to output 2 classes.
        """
        super().__init__()

        # Download the weights
        self.weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        # Load the model with the pretrained weights
        self.model = torchvision.models.resnet18(
            weights=self.weights,
            progress=True,
        )
        # Freeze the all layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Change the classifier layer to output 2 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

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


class ResNet34(nn.Module):
    """Class that returns a ResNet34 model with the last layer changed to output 2 classes.

    Attributes:
        weights (str): Downloaded weights for the ResNet34 model from IMAGENET1K_V1.
        model (torch.nn.Module): ResNet18 model with the pretrained weights on all layers.

    Methods:
        forward: Passes the input tensor x through the model.
    """

    def __init__(
        self: Self,
    ) -> None:
        """Download the IMAGENET1K_V1 weights for ResNet34 and initializes the model with 
        the pre-trained weights on all layers. Afterwards, freeze all layers except the last
        one and change it to output 2 classes.
        """
        super().__init__()

        # Download the weights
        self.weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        # Load the model with the pretrained weights
        self.model = torchvision.models.resnet34(
            weights=self.weights,
            progress=True,
        )
        # Freeze the all layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Change the classifier layer to output 2 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

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
