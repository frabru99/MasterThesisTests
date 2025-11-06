""" Central configuration file"""

import torch 

# ---- Model & Training Settings ----
NUM_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = "cpu"

# ---- File Paths ----
DATA_DIR = "../../TestDatasets/casting_data"

# ---- Fine tuned model path ---
MODEL_PATH = "./casting_efficientnet_b0.pth"

# ---- ONNX model path ----
ONNX_MODEL_PATH = "./casting_efficientnet_b0.onnx"

# --- Image Settings -----
IMAGE_SIZE = 224
