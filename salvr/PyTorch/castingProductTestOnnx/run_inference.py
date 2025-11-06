""" Module to run inference using the exported ONNX model. """

import onnxruntime as ort
import numpy as np
import torch 

from PIL import Image

import config
from data_loader import get_dataloaders, get_model_transforms

def preprocess_image_onnx(image_path, transform):

    """
    Loads and preprocesses a single image for ONNX inference
    """

    img = Image.open(image_path) # CHECK RGB THING!!!!!
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t.numpy()

def main():

    print(f"Loading ONNX model from {config.ONNX_MODEL_PATH}")
    try:
        ort_session = ort.InferenceSession(config.ONNX_MODEL_PATH)
        input_name = ort_session.get_inputs()[0].name
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        print("Did you run export_to_onnx.py first?")
        return
    
    # Test on a single sample image
    print("\n----- Single Image Test ----")

    # Get the test transforms
    inference_transform = get_model_transforms()['test']
    sample_image_path = f"{config.DATA_DIR}/test/def_front/cast_def_0_7.jpeg"

    try:
        input_data = preprocess_image_onnx(sample_image_path, inference_transform)
        
        # Run ONNX inference
        outputs = ort_session.run(None, {input_name: input_data})
        onnx_prediction = outputs[0]
        
        # Interpret results
        predicted_index = np.argmax(onnx_prediction)
        
        # We need the class names
        _, class_names = get_dataloaders()
        
        predicted_class = class_names[predicted_index]
        print(f"Sample: {sample_image_path}")
        print(f"ONNX Model Prediction: {predicted_class}")
        
    except FileNotFoundError:
        print(f"Sample image not found at {sample_image_path}")
    except Exception as e:
        print(f"Error during single image test: {e}")

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
    print(f"Accuracy of ONNX model on test set: {accuracy:.2f} %")

if __name__ == "__main__":
    main()



