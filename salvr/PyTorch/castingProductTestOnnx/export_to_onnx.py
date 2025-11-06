""" Module to convert the fine-tuned PyTorch model to ONNX """

import torch

import config
from model_trainer import get_model 

def main():
    print("Exporting model to ONNX....")

    # Get the model structure
    model = get_model(num_classes=2)

    # Loading the saved model
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH))
    except Exception as e:
        print(f"Error loading model weights from {config.MODEL_PATH}")
        print("Did you run model_trainer.py first?")
        return

    model.to(config.DEVICE)
    model.eval() # Set to evaluation mode

    # create a dummy input for tracing
    dummy_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        config.ONNX_MODEL_PATH,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names = ['input'],
        output_names = ['output'],
        dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
    )

    print(f"Model successfully exported to {config.ONNX_MODEL_PATH}")

if __name__ == "__main__":
    main()
