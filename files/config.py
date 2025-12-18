from pathlib import Path

# Directory containing the dataset
DATA_DIR = Path(__file__).parent.parent / "data" / "clothes_dataset" / "dataset"

# Base project directory
BASE_DIR = Path(__file__).parent.parent 

# Directory to save plots
SAVE_DIR = BASE_DIR / "plot"
SAVE_DIR.mkdir(parents=True, exist_ok=True) 

# Directory to save trained model
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True) 

# Input image shape for the network
Image_shape = (96, 96, 3)

# Number of training epochs
EPOCHS = 40

# Verbosity level for training output
VERBOS = 1