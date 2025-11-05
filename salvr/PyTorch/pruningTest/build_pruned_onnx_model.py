# Using a classic MobileNetV2, but pruned, to make inferences in onnx format

import torch
import torchvision
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from pruning import simple_pruning_fn, pruning_multiple_parameters

# Number of classes
NUM_CLASSES = 10 

# Loading the model and set evaluation mode
model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, NUM_CLASSES)
model.to("cpu")
model.eval()
print("PyTorch model loaded and modified")

# Applying the previous pruning function
print("Applying the pruning of multiple parameters...")
pruning_multiple_parameters(model)
print("Pruning applied!")

# Making the pruning permanent
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        if prune.is_pruned(module):
            prune.remove(module, "weight")
print("Pruning hooks removed. Model is now permanently sparse")

# Exporting the model in onnx
model_name = "mobilenetV2_pruned_for_inference"
dummy_input = torch.randn(1, 3, 224, 224)

print(f"Exporting permanently pruned model to {model_name}.onnx")
torch.onnx.export(
    model,
    dummy_input,
    f"training_artifacts/{model_name}.onnx",
    input_names=["input"],
    output_names=["output"],
    export_params=True,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    do_constant_folding=True,
    training=torch.onnx.TrainingMode.EVAL,
    opset_version=14,
    dynamo=False
)

print("Export complete!")