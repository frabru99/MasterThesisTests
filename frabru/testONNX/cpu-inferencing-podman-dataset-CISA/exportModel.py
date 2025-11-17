import torchvision.models as models
import torch.nn as nn
import torch
import os
from typing import Optional
from modelConfig import ModelConfig


WEIGHTS_PATH = "models/final_model.pth" 
MODEL_NAME = "mobilenet_v2_trained" 


def load_model() -> nn.Module:
    
    print("Initializing MobileNetV2 model structure...")
    
    model = models.mobilenet_v2(weights=None)


    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Linear(in_features, len(ModelConfig.classes)))
    
    print(f"Loading weights from: {WEIGHTS_PATH}")
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights file not found at: {WEIGHTS_PATH}")

    # Load the weights
    state_dict = torch.load(WEIGHTS_PATH, map_location=ModelConfig.device)

    #Check if the .pth file is a full checkpoint (e.g., contains 'model_state_dict')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        print("Detected checkpoint structure and extracted 'state_dict'.")
    elif isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
        print("Detected checkpoint structure and extracted 'model_state_dict'.")


    #print(f"LOADED STATE DICT: {state_dict}")
    # Load the state dictionary into the model
    # strict=True means all keys in the state_dict must match the model's keys
    model.load_state_dict(state_dict, strict=True) 
    print("Model weights loaded successfully.")
    
    # Set the model to evaluation mode for consistent export
    model.eval() 
    
    return model


def exportModelONNX(model: nn.Module, model_name: str):
    """
    Exports the loaded PyTorch model to the ONNX format.
    """

    model.eval()

    print(f"\nStarting ONNX export for {model_name}...")
    
    ARTIFACTS_DIR = "artifacts"
    
    if not os.path.exists(ARTIFACTS_DIR):
        os.mkdir(ARTIFACTS_DIR)

    output_path = f"./{ARTIFACTS_DIR}/{model_name}.onnx"

    # Dummy input: Batch=1, Channels=3, Height=224, Width=224 (St
    # andard for MobileNetV2)
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    
    # Use a torch.no_grad() context manager to be safe, especially if model is not set to model.eval()
    torch.onnx.export(model, 
                        dummy_input, 
                        output_path, 
                        input_names=["input"], 
                        output_names=["output"], 
                        export_params=True, 
                        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                        do_constant_folding=True, 
                        # Set training mode to EVALUATION (default is TRAINING) for deployment.
                        # Since you have `model.eval()` above, this is redundant but safer.
                        training=torch.onnx.TrainingMode.EVAL, 
                        opset_version=13, 
                        dynamo=False)

    print(f"\n Model successfully exported to: {output_path}")


    