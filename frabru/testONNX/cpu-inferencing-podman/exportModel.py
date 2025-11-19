import torch
import torchvision
import os

NUM_CLASSES=10 #Num. of classes of the new classification task. 

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1) #Download the pre-trained model on IMAGENET with 1K classes. 


num_ftrs = model.fc.in_features #Takes the number of inputs of the FC layer 

model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES) #substitution of the FC layer with a new one, with the same number of input but a new number of output classes. 


model_name = "resnet" 

model.train() #Training Mode


os.mkdir("artifacts")


torch.onnx.export(model, 
				  #Dummy input for graph generation (Batch=1, Channels=3 and dimensiond)
                  torch.randn(1,3,224,224), 
                  
                  #name of the output file
                  f"./artifacts/{model_name}.onnx", 
                  input_names=["input"], #symbolic name for input
                  output_names=["output"],  #symbolic name for output
                  
                  export_params=True, #exports bias and weights in onnx file
                 
                  #Setting of the dynamic axes, input and output dimension are
                  #substituted with parameters in order to use various batch
                  #sizes
                  dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}},
                  
                  #No Constant Folding Optimization (often used in TRAINING)
                  do_constant_folding=False, 
                  
                  #Setting onnx model for training
                  training = torch.onnx.TrainingMode.TRAINING, 
                  
                  opset_version=13, #Specifies ONNX Operator Set's Version
                  
                  dynamo=False #Disables torch.dynamo in order to use 
                  # legacy solution
                  )
