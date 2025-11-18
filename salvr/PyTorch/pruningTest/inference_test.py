
import torch
import torchvision
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from pruning import simple_pruning_fn, pruning_multiple_parameters

try: 
    import onnxruntime as ort
    import numpy as np
except ImportError:
    print("Please install onnxruntime: pip install onnxruntime")
    exit()

# Enabling profiling in session_options to create a json file with the profiling times
sess_options = ort.SessionOptions()
sess_options.enable_profiling = True

model_name = "mobilenetV2_pruned_for_inference"
dummy_input = torch.randn(1, 3, 224, 224)
onnx_model_path = f"training_artifacts/{model_name}.onnx"

# Create an inference session with onnx_runtime
print(f"Loading ONNX model from {onnx_model_path}")
session = ort.InferenceSession(onnx_model_path, sess_options= sess_options, providers = ['CPUExecutionProvider'])


input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Model Input: {input_name}")
print(f"Model Output: {output_name}")

# We can re-use the 'dummy_input' variable from the export step
input_data = dummy_input.numpy()

print(f"Runnig inference with input shape: {input_data.shape}")

# ort.InferenceSession.run() expects a list of outputs and a dictionary of inputs
results = session.run([output_name], {input_name: input_data})

# 'results' will be a list containing one element (our output array)
output_predictions = results[0]

print("\n--- Inference Successful! ---")
print(f"Output shape: {output_predictions.shape}")
print("Output predictions (first 5 values):")
print(output_predictions[0, :5])

# To get the final predicted class
predicted_class = np.argmax(output_predictions, axis=1)
print(f"Predicted class index: {predicted_class[0]}")

# Call the end profiling
profiler_file = session.end_profiling()


