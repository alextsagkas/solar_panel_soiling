import torchvision
from torch import nn

# Get the weights
weights_resnet18 = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
# Load the weights to ResNet18
ResNet18 = torchvision.models.resnet18(
    weights=weights_resnet18,
    progress=True,
)
# Freeze the Layers
for param in ResNet18.parameters():
    param.requires_grad = False
# Change the Classifier layer to output 2 classes
ResNet18.fc = nn.Linear(ResNet18.fc.in_features, 2)
