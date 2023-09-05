import torchvision

weights_resnet18 = torchvision.models.ResNet18_Weights.DEFAULT
Resnet18 = torchvision.models.resnet18(weights=weights_resnet18)
for param in Resnet18.parameters():
    param.requires_grad = False
