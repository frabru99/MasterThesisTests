""" Module to run inference using the exported ONNX model. """

import onnxruntime as ort
import numpy as np
import torch 
import os

from typing import Union
from PIL import Image

import config
from data_loader import get_dataloaders, get_model_transforms

def measure_onnx_weight(model_path: str, unit: str="MB") -> Union[float, int]:

    # Size bytes in os.path.getsize are returned in byte
    try:
        size_bytes = os.path.getsize(model_path)
    except FileNotFoundError:
        print(f"Error: ONNX file not found at {model_path}")
        return 0
    
    # Changing the unit with the one choosen
    unit = unit.upper()
    if unit == "KB":
        return size_bytes / (1024)
    elif unit == "MB":
        return size_bytes / (1024 * 1024)
    elif unit == "GB":
        return size_bytes / (1024 * 1024 * 1024)
    else:
        return size_bytes
    

def preprocess_image_onnx(image_path, transform):

    """
    Loads and preprocesses a single image for ONNX inference
    """

    img = Image.open(image_path)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t.numpy()

def run_inference_on_casting(model_method: str):

    current_onnx_path = ""
    model_paths = [
        config.ONNX_MODEL_PATH,
        config.ONNX_PRUNED_MODEL_PATH
    ]

    for path in model_paths:
        if model_method in path:
            current_onnx_path = path
            break

    print(f"Loading ONNX model from {current_onnx_path}")
    try:
        ort_session = ort.InferenceSession(current_onnx_path)
        input_name = ort_session.get_inputs()[0].name
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        print("Did you run export_to_onnx.py first?")
        return

    # Run inference on the whole test set
    print("\n--- Full Test Set Evaluation ---")
    dataloaders, class_names = get_dataloaders()
    test_loader = dataloaders['test']
    
    correct = 0
    total = 0

    for inputs, labels in test_loader:
        input_numpy = inputs.cpu().numpy()
        labels_numpy = labels.cpu().numpy()

        # Run Onnx inference
        onnx_outputs = ort_session.run(None, {input_name: input_numpy})[0]

        # Get predicted indices
        predicted_indices = np.argmax(onnx_outputs, axis=1)

        # Compare with true labels
        total += labels_numpy.shape[0]
        correct += (predicted_indices == labels_numpy).sum().item()

    accuracy = 100 * correct / total
    unit = "KB"
    model_weight = measure_onnx_weight(current_onnx_path, unit)
    print(f"Accuracy of ONNX {model_method} on test set: {accuracy:.2f} %")
    print(f"Weight of ONNX {model_method} is {model_weight} {unit}")

def main():
   
   print(torch.__version__)

   run_inference_on_casting("Base")
   run_inference_on_casting("Pruned") 
    


if __name__ == "__main__":
    main()



