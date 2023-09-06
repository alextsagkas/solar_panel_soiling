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
debug_dir = root_dir / "debug"

tensorboard_dir = debug_dir / "runs"
test_model_dir = debug_dir / "test_model"
metrics_dir = debug_dir / "metrics"
data_transforms_dir = debug_dir / "data_transforms"
checkpoint_dir = debug_dir / "checkpoints"
