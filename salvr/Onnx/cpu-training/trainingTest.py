import numpy as np
import onnxruntime.training.api as onnxtraining
from onnxruntime.training.api import CheckpointState, Module, Optimizer
import onnxruntime as ort
import torch
import torchvision
import torchvision.transforms as transforms
import os

BATCH_SIZE = 60 #Batch Size
NUM_CLASSES = 10 #Same as Classes of the Dataset
ARTIFACT_DIR = "./training_artifacts"

print("Device Choice...")

#device_type= "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
device_type = "cpu"
print(f"Using device: {device_type}" )


print("Loading CIFAR-10...")


# Input Images resizing for Resnet (224, 224)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Download the Dataset 
train_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                             train=True,
                                             download=True, 
                                             transform=transform)

# Data Loader for batch generation
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
print("Dataset ready.")


print("Loading ONNX artifacts...")

try:
    state = onnxtraining.CheckpointState.load_checkpoint(f"{ARTIFACT_DIR}/checkpoint")
    model = Module(f"{ARTIFACT_DIR}/training_model.onnx", state, device=device_type)
    optimizer = Optimizer(f"{ARTIFACT_DIR}/optimizer_model.onnx", model)
except Exception as e:
    print(f"Error in loading artifacts from {ARTIFACT_DIR}.")
    print("Make sure to extract the correct model with right number of classes in FC layer.")
    print(f"Error Details: {e}")
    exit()

print("Model loaded. Training Loop Starts...")

# Training Loops 
for i, (images_tensor, labels_tensor) in enumerate(train_loader):

    # Images in numpy array
    images = images_tensor.numpy()
    
#### FOR BCEWithLogitsLoss, ONE-HOT ENCODING
    # --- 2. Prepara le Etichette (One-Hot Encoding) ---
        
    # 2a. Ottieni gli indici (es. [3, 0, 8, 1])
    #labels_indices = labels_tensor.numpy().astype(np.int64)
    
    # 2b. Gestisci l'ultimo batch (potrebbe non essere pieno)
    #current_batch_size = labels_indices.shape[0]
    
    # 2c. Crea una matrice di zeri (es. 4x10)
    #labels_one_hot = np.zeros((current_batch_size, NUM_CLASSES), dtype=np.float32)
    
    # 2d. Popola '1.0' nelle posizioni corrette
    # Questo converte [3, 0, 8, 1] in [[0,0,0,1,0..], [1,0,0,0,0..], ...]
    #labels_one_hot[np.arange(current_batch_size), labels_indices] = 1.0
    
    # 2e. 'labels' è ora l'array NumPy 2D [float32] corretto
    #labels = labels_one_hot
    # --- Fine blocco etichette ---

    # Labels in numpy array, in int64 type
    labels = labels_tensor.numpy().astype(np.int64)

    # TRAINING STEP: Forward, Loss and Backword propagation
    loss = model(images, labels)

    # Applies the gradients on weights and biases
    optimizer.step()

    # Gradients reset
    model.lazy_reset_grad()

   
    if i % 10 == 0: # Every 10 batches
        print(f"Step {i}, Loss: {loss.item():.4f}")

print("Training complete.")

try:
    CheckpointState.save_checkpoint(state, f"{ARTIFACT_DIR}/checkpoint_updated")
    print(f"Checkpoint updated in {ARTIFACT_DIR}/checkpoint_updated")
except Exception as e:
    print(f"Error in checkpoint updated saving: {e}")
