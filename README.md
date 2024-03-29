# Solar Panel Soiling Detection

| Model Name         | Test Accuracy (%) | Config File                      |
| ------------------ | ----------------- | -------------------------------- |
| ShuffleNet_V2_X1_5 | 93.97             | 2023-09-20/16-29-21_epoch_79.txt |
| ResNet34           | 91.00             | 2023-09-12/12-42-18_epoch_11.txt |
| ShuffleNet_V2_X2_0 | 90.36             | 2023-09-15/12-14-30_epoch_1.txt  |
| EfficientNet_B2    | 89.30             | 2023-09-11/16-45-38_epoch_21.txt |
| EfficientNet_V2_M  | 88.89             | 2023-09-11/19-25-44_epoch_23.txt |
| EfficientNet_V2_L  | 88.50             | 2023-09-12/01-00-32_epoch_1.txt  |
| MobileNet_V3_Large | 88.45             | 2023-09-13/00-49-38_epoch_28.txt |
| EfficientNet_B7    | 88.30             | 2023-09-12/19-13-46_epoch_6.txt  |
| MobileNet_V3_Small | 87.65             | 2023-09-12/23_18_25_epoch_9.txt  |
| EfficientNet_B0    | 87.34             | 2023-09-11/13-19-26_epoch_27.txt |
| EfficientNet_B1    | 86.90             | 2023-09-11/16-04-08_epoch_9.txt  |
| ResNet50           | 86.02             | 2023-09-12/15-16-40_epoch_2.txt  |
| ResNet18           | 85.79             | 2023-09-12/12-10-22_epoch_9.txt  |
| EfficientNet_B3    | 85.77             | 2023-09-11/01-46-03_epoch_5.txt  |
| MobileNet_V2       | 85.51             | 2023-09-12/22-01-44_epoch_14.txt |
| EfficientNet_B6    | 83.73             | 2023-09-11/17-16-42_epoch_8.txt  |
| EfficientNet_V2_S  | 83.78             | 2023-09-11/19-01-14_epoch_4.txt  |
| ShuffleNet_V2_X1_0 | 81.80             | 2023-09-15/17-05-21_epoch_60.txt |
| ShuffleNet_V2_X0_5 | 79.57             | 2023-09-15/01-00-49_epoch_11.txt |

## Requirements & Installation

The project is developed in **Python 3.9.17**. The required packages are listed in the `requirements.txt` file. To create a virtual environment and to install them, run the following commands:

```bash
python3.9 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

More information can be found in [freecodecamp.org](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/).

## Data

The data used in this project is from [deep-solar-eye](https://deep-solar-eye.github.io). They are separated in two folders: `data/train/` and `data/test/`. The `data/train/` folder contains 80% of the data and the `data/test/` folder the remaining 20%. They are used for training of the neural network and testing it, respectively.

Image file names contain the time of the day, power loss of the panel with respect to the clean panel and irradiance level. For example, the file name can be parsed as:

- `solar_Wed_Jun_28_7__5__6_2017_L_0.0123268698061_I_0.0566274509804.jpg`
- `solar_day_Month_date__hour__minute__second_year_L_loss_I_irradiancelevel.jpg`

In addition, another, tiny in comparison, dataset have been constructed of random images from Google located in the `data/results/` folder. These images are used to evaluate the model in real world images.

## How to Run the Codebase

The codebase have been organized so as the user to interact **only** with the `packages/main.py` file. Precisely, in the beginning of the file, there are tests the user can run (they are located in `packages/tests/` folder). When the `test_name` is chosen, the user can modify the configuration of each test by altering the `hyperparameters` dictionary inside the corresponding `if` statement. What each test does is described in its docstring.

More information about how the project is structured and tied together can be found in the following sections.

## Hyperparameters

Each test run results in saving the hyperparameters used to configure it in `config/YYYY-DD-MM/HH-MM-SS.txt`, where `YYYY-DD-MM/HH-MM-SS` is the date and time the test was run. The hyperparameters are saved in a line by line format so as to be able to reconstruct the hyperparameter dictionary while loading them.

The explanation of this choice is that the user can easily reproduce the results of a test by copying the hyperparameters from the file and pasting them in the `hyperparameters` dictionary of the corresponding test in `packages/main.py`. This way, the user can easily modify the hyperparameters and run the test again.

Additionally, a trained model can be easily loaded again and evaluated on a random test set by just locating its config file. This is useful when the user wants to train a multiplicity of models and use the best one for testing.

The same convention of `YYYY-DD-MM/HH-MM-SS/` is used to save every single information about the current test. Precisely, the model is saved in `models/YYYY-DD-MM/HH-MM-SS.pth`, the results of the test in `debug/metrics/YYYY-DD-MM/HH-MM-SS.txt` and the tensorboard logs in `debug/runs/YYYY-DD-MM/HH-MM-SS/`.

All the functionality for loading and saving the hyperparameters is located in `packages/utils/storage.py`.

## Global Paths

The global paths used to store and load data in the project are located in `packages/utils/configuration.py`. Any change in this file correctly updates the paths in the project.

## Models

The models used in this project are implemented in `packages/models/` folder as a `torch.nn.Module` class. In order to load use them the `packages/utils/models.py` file is used. It contains the `GetModels` class which picks a model by its name (the name of the model corresponds to the corresponding method in the class).

There is also support for configuring certain properties of each model through the optional `config` dictionary attribute. For example, in order to overwrite the default `hidden_units` the following `config` can be used:

```python
config = {
    "hidden_units": 512
}
```

Every model supports different configuration, so passing the config parameter can be tricky sometimes. Review the documentation of the class and match the models attributes so as to pass the correct config.

### Training

The training functionality is encapsulated in the Solver class located in the `packages/utils/solver.py`. It provides two options for training.

The simple training methods trains the model for a number of epochs. Then, the trained model is tested on the test dataset and the results are saved in a file. The training process is logged in tensorboard.

The k-fold cross validation method firs splits the train dataset to k folds. Then, it trains the model k times, each time using a different fold as validation set, and all the remaining as the test set. The results of each fold are saved in a file. At the end of each fold the model is tested on the test dataset and then saved. The training process is logged in tensorboard.

More information about the training process can be found in the docstring of the Solver class.

### Testing - Inference

The testing functionality is encapsulated in the `packages/utils/tester.py`. This function is used on the smaller dataset of random images from Google. Except from testing the model, it also saves the images it tested and with which probability it classified them. The images are saves in `debug/test_model/YYYY-DD-MM/HH-MM-SS/` folder and the results in `debug/metrics/YYYY-DD-MM/HH-MM-SS.txt` file.

## Loss Function & Class Probabilities

Using Cross Entropy Loss function for the binary classification of images in two classes. Therefore, the
model outputs two logits, one for each class. The probabilities for each class are calculated using the
softmax function.

## Data Augmentation

All the functionality for transforming the data is located in a single class in `packages/utils/transforms.py`. The class is called `GetTransform` and works similarly with the `GetModels` class. It picks a transform by its name (the name of the transform corresponds to the corresponding method in the class) and configures it through the optional `config` dictionary.

## Optimizers

In order to update the parameters of a model an optimizer is used. The optimizers are accessed through `GetOptimizer` class located in `packages/utils/optim.py`. It picks an optimizer by its name (the name of the optimizer corresponds to the corresponding method in the class) and configures it through the optional `config` dictionary. Also, it needs the parameters of the model to be optimized.
