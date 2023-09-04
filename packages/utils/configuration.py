from pathlib import Path

# Root
root_dir = Path("/Users/alextsagkas/Document/Office/solar_panels")

# Data
data_dir = root_dir / "data"
train_dir = data_dir / "train"
test_dir = data_dir / "test"
results_dir = data_dir / "results"

# Models
models_dir = root_dir / "models"

# Configuration
config_dir = root_dir / "config"

# Debug
tensorboard_dir = root_dir / "debug" / "runs"
test_model_dir = root_dir / "debug" / "test_model"
metrics_dir = root_dir / "debug" / "metrics"
data_transforms_dir = root_dir / "debug" / "data_transforms"
