import os
import argparse


parser = argparse.ArgumentParser(description="Get hyper-parameters")

# Get an arg for num_epochs
parser.add_argument("--num_epochs",
                    default=10,
                    type=int,
                    help="the number of epochs to train for")

# Get an arg for batch_size
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="number of samples per batch")

# Get an arg for hidden_units
parser.add_argument("--hidden_units",
                    default=10,
                    type=int,
                    help="number of hidden units in hidden layers")

# Get an arg for learning_rate
parser.add_argument("--learning_rate",
                    default=0.001,
                    type=float,
                    help="learning rate to use for model")

# Create an arg for training directory
parser.add_argument("--train_dir",
                    default="data/pizza_steak_sushi/train",
                    type=str,
                    help="directory file path to training data in standard image classification format")

# Create an arg for test directory
parser.add_argument("--test_dir",
                    default="data/pizza_steak_sushi/test",
                    type=str,
                    help="directory file path to testing data in standard image classification format")

# Get our arguments from the parser
args = parser.parse_args()
# Write them on config.txt file
with open("config.txt", "w") as f:
    f.write(args.__str__())

# Setup hyper-parameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
print(
    f"[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} using {HIDDEN_UNITS} hidden units and a learning rate of {LEARNING_RATE}")

# Setup directories
train_dir = args.train_dir
test_dir = args.test_dir
print(f"[INFO] Training data file: {train_dir}")
print(f"[INFO] Testing data file: {test_dir}")
