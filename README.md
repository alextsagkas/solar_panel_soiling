# Solar Panel Soiling Detection

## Virtual Environment

Create a virtual environment with python3.9 and install the packages in requirements.txt file. More information can be found in [freecodecamp.org](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/).

## Data

The data used in this project is from [deep-solar-eye](https://deep-solar-eye.github.io). They are separated in two folders: `data/train/` and `data/test/`. The `data/train/` folder contains 80% of the data and the `data/test/` folder the remaining 20%. They are used for training of the neural network and testing it, respectively.

Image file names contain the time of the day, power loss of the panel with respect to the clean panel and irradiance level. For example, the file name can be parsed as:

- `solar_Wed_Jun_28_7__5__6_2017_L_0.0123268698061_I_0.0566274509804.jpg`
- `solar_day_Month_date__hour__minute__second_year_L_loss_I_irradiancelevel.jpg`

## Loss Function & Class Probabilities

Using Cross Entropy Loss function for the binary classification of images in tow classes. Therefore, the
model outputs two logits, one for each class. The probabilities for each class are caclulated using the
softmax function.
