import onnx
from onnxruntime.training import artifacts

#loading the model from .onnx file
onnx_model: onnx.ModelProto = onnx.load("./training_artifacts/resnet.onnx")

#all initializers: weights and bias. 
all_initializers = {init.name for init in onnx_model.graph.initializer}

print(f"All Initializers: {len(all_initializers)}")

trainable_param = []

#Find trainable params for "require_grad" parameter
for param_name in all_initializers:
    if not (param_name.endswith(".running_mean") or param_name.endswith(".running_var")): 
        trainable_param.append(param_name)

print(f"Trainable params: {len(trainable_param)}")


artifacts.generate_artifacts(
    onnx_model,  
    requires_grad=trainable_param,
    loss=artifacts.LossType.CrossEntropyLoss, #Loss Choose
    optimizer=artifacts.OptimType.SGD, #Optimizer choose
    artifact_directory="./training_artifacts" #artifact directory
)
