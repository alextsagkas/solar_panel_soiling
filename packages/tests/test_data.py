from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from packages.utils.load_data import plot_transformed_images

# Get image paths
image_path = Path("/Users/alextsagkas/Document/Office/solar_panels/data/")
train_dir = image_path / "train"
test_dir = image_path / "test"
image_path_list = list(image_path.glob("*/*/*.jpg"))

# Create image transform
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5),  # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor()  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])

# Load data
train_data = datasets.ImageFolder(root=str(train_dir),  # target folder of images
                                  transform=data_transform,  # transforms to perform on data
                                  target_transform=None)  # transforms to perform on labels

test_data = datasets.ImageFolder(root=str(test_dir),
                                 transform=data_transform)

# Create data loaders
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=1,  # how many samples per batch?
                              num_workers=1,  # how many subprocesses to use for data loading
                              shuffle=True)  # shuffle the data

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=1,
                             num_workers=1,
                             shuffle=False)  # don't usually need to shuffle testing data

class_names = train_data.classes
class_dict = train_data.class_to_idx

if __name__ == "__main__":
    # Visualize  the results
    plot_transformed_images(image_path_list, data_transform, n=10, seed=42)
