
from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

import numpy as np
import onnxruntime as ort
import onnx
import torch
from importlib import import_module
from pathlib import Path
from torchvision import models
from BenchmarkingFactory.dataWrapper import DataWrapper


PROJECT_ROOT = Path(__file__).resolve().parent.parent

class AIModel():

    def __init__(self, model_info: dict):
        
        """
        Constructor to create an AIModel used in a DoE experiment

        Input:
            - model_info: a disctionary from the configuratro module (already checked)
            it has the following example structure

            dic = {'resnet50': {
            'module': 'torchvision.models',
            'model_name',
            'native': True,
            'distilled': False,
            'weights_path': "",
            'device': "cpu",
            'class_name': 'resnet50',
            'weights_class': 'ResNet50_Weights',
            'image_size': 224,
            'description': 'ResNet-50 from torchvision'
                }
            }
        """

        self.model_info = model_info
        self.model = self._loadModel(model_info['module'], model_info['class_name'])


    def getInfo(self, info_name: str):

        return self.model_info[info_name]

    def _replaceModelClassifier(self, model, class_name):
        """
        Function that automatically recognize the final layer and replace for the inference on a specific dataset

        Inputs:
            - model: Model passed that needs to be recognized
        """

        # Inferencing the classifier of the model, changing it
        if self.getInfo('num_classes') is not None:
            last_layer_name, last_layer = None, None

            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    last_layer_name = name
                    last_layer = module

            if last_layer is None:
                logger.error(f"Could not find a Linear layer to replace in {class_name}")
            else:
                # Get in_features and create new layer
                try:
                    in_features = last_layer.in_features
                    num_classes = self.getInfo('num_classes')
                    new_classifier = torch.nn.Linear(in_features, num_classes)
                    
                    # Split the name to navigate it and replace last layer
                    parts = last_layer_name.split('.')
                    if len(parts) == 1:
                        # Simple case: 'fc'
                        setattr(model, last_layer_name, new_classifier)
                    else:
                        # Nested case: 'classifier.1' or heads.head
                        parent_module = model
                        for part in parts[:-1]:
                            # Handle cases where parent is a Sequential list
                            if part.isdigit():
                                parent_module = parent_module[int(part)]
                            else:
                                parent_module = getattr(parent_module, part)
                    
                        # Set the last part
                        last_part_name = parts[-1]
                        if last_part_name.isdigit():
                            parent_module[int(last_part_name)] = new_classifier
                        else:
                            setattr(parent_module, last_part_name, new_classifier)
                    
                    logger.debug(f"Replace {last_layer_name} for {num_classes} classes.")

                except Exception as e:
                    logger.error(f"Failed to auto-replace classifier for {class_name}: {e}")

    def _loadModel(self, module, class_name): 
        """
        Function to reconstruct the model from module and the class name

        Input: 
            - module: Module from PyTorch Library (e.g., torchvision.models)
            - class_name: Attribute name of the module, to load the model structure
        """
        
        module = import_module(module)
        model_class = getattr(module, class_name)
        model = model_class()

        # Try to replace last layer classifier
        self._replaceModelClassifier(model, model_class)

        try:

            weights = self.getInfo('weights_path')
            model_checkpoint = torch.load(weights)

            # Check if it's a checkpoint dictionary or a raw state_dict
            if 'model_state_dict' in model_checkpoint:
                model.load_state_dict(model_checkpoint['model_state_dict'])
                logger.debug(f"Loaded Weights from model_state_dict key in {weights}")
            
            # Assuming it's a raw state_dict
            else:
                model.load_state_dict(model_checkpoint)
                logger.debug(f"Loaded Weights raw state_dict from {weights}")

        except FileNotFoundError:
            logger.error(f"Weight file not found at {self.model_info['weights_path']}. Model has random weights.")
        except Exception as e:
            logger.error(f"Error loading model weights from {self.model_info['weights_path']}: {e}")

        return model

    def _getProviderList(self, device: str) -> list:
        """
        Function that scans possible providers and give the right provider list for the inference session

        Input: 
            - device: a string given from the configurator module (e.g., 'cpu' or 'gpu')

        Output:
            - provider_list: list of possible providers ordered in terms of priority
        """

        machine_provider_list = ort.get_available_providers()
        choosen_device_provider_list = []


        if device == "cpu":
            choosen_device_provider_list.extend([
                "OpenVINOExecutionProvider", 
                "ArmNNExecutionProvider", 
                "ACLExecutionProvider"
            ])
        elif device == "gpu":
            choosen_device_provider_list.extend([
                "TensorRTExecutionProvider", 
                "CUDAExecutionProvider", 
                "ROCMExecutionProvider"
            ])

        
        # Always put the cpu provider at the end as a fallback
        choosen_device_provider_list.append("CPUExecutionProvider")

        # Intersect the wanted providers with actual available providers
        final_provider_list = [
            provider for provider in choosen_device_provider_list 
            if provider in machine_provider_list
        ]

        if not final_provider_list:
            logger.warning(f"Provider not correctly choosed, setting default cpu provider")
            return ["CPUExecutionProvider"]

        return final_provider_list

    def getModel(self):
        """
        Returns the torch model attached to aimodel
        """

        return self.model

    def createOnnxModel(self, input_data):
        """
        Function that creates the onnx file to use for the inference part
        """

        device = self.getInfo('device')
        self.model.to(device)
        self.model.eval()

        batch_dim = torch.export.Dim("batch_size")

        image_size = self.getInfo('image_size')

        dummy_input, _ = next(iter(input_data))
        dummy_input = dummy_input.to(device)

        model_name = self.getInfo('model_name')
        onnx_model_path = PROJECT_ROOT / "ModelData" / "ONNXModels" / f"{model_name}.onnx"

        dynamic_shapes_config = {
            "x": {0: batch_dim}
        }

        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_model_path,
            export_params=True,
            opset_version=18,
            do_constant_folding = True,
            input_names = ['input'],
            output_names = ['output'],
            dynamic_shapes = dynamic_shapes_config
        )

    def runInference(self, input_data):
        """
        Function to run inference on a given dataset

        Inputs: 
            - input_data: Dataset passed from the dataloader
        """

        device_str = self.getInfo('device')
        model_name = self.getInfo('model_name')
        onnx_model_path = PROJECT_ROOT / "ModelData" / "ONNXModels"  /  f"{model_name}.onnx"

        provider_list = self._getProviderList(self.model_info['device'])

        device_name = "cuda" if device_str == "gpu" else "cpu"

        try:
            ort_session = ort.InferenceSession(str(onnx_model_path), providers=provider_list)
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            
            input_type = np.float32
            output_type = np.float32

            logger.debug(f"Input of ort session: {ort_session.get_inputs()[0]}")
            logger.debug(f"Output of ort session: {ort_session.get_outputs()[0]}")

        except Exception as e:
            logger.debug(f"Error loading ONNX model: {e}")

        logger.debug(f"--- Full test set evaluation ---")
        io_binding = ort_session.io_binding()

        total = 0
        correct = 0

        with torch.no_grad():
            for inputs, labels in input_data:

                labels = labels.to(device_str)
                batch_size = inputs.shape[0]

                if device_name == "cuda":

                    # Moving the input on the same device of the onnx session for zero copy
                    input_tensor = inputs.to(device_str).contiguous()

                    # Pre allocate output tensor on gpu
                    output_shape = (batch_size, output_num_classes)
                    onnx_outputs_tensor = torch.empty(output_shape, 
                                                      dtype=torch.float32, 
                                                      device=device_name).contiguous()

                    # Binding Input and Outputs
                    io_binding.bind_input(
                        name=input_name,
                        device_type='cuda',
                        device_id=0,
                        element_type=input_type,
                        shape=tuple(inputs_tensor.shape),
                        buffer_ptr=inputs_tensor.data_ptr()
                    )
                    io_binding.bind_output(
                        name=output_name,
                        device_type='cuda',
                        device_id=0,
                        element_type=output_type,
                        shape=tuple(onnx_outputs_tensor.shape),
                        buffer_ptr=onnx_outputs_tensor.data_ptr()
                    )

                    # Run inference with binding options
                    ort_session.run_with_iobinding(io_binding)

                elif device_name == "cpu":

                    input_as_numpy = inputs.numpy()

                    ort_input_value = ort.OrtValue.ortvalue_from_numpy(input_as_numpy)

                    # Binding inputs and outputs on cpu
                    io_binding.bind_input(
                        name=input_name,
                        device_type='cpu',
                        device_id=0,  
                        element_type=input_type,
                        shape=tuple(input_as_numpy.shape),
                        buffer_ptr=input_as_numpy.ctypes.data 
                    )
                    io_binding.bind_output(output_name, device_type = 'cpu',
                                            device_id=0)

                    ort_session.run_with_iobinding(io_binding)

                    # Get outputs and reconvert into torch tensors
                    onnx_outputs_ort = io_binding.get_outputs()
                    numpy_output = onnx_outputs_ort[0].numpy()
                    onnx_outputs_tensor = torch.from_numpy(numpy_output)

                # Cleaning binding for next iteration
                io_binding.clear_binding_inputs()
                io_binding.clear_binding_outputs()

                predicted_indices = torch.argmax(onnx_outputs_tensor, dim=1)

                total += labels.shape[0]
                correct += (predicted_indices == labels).sum().item()

            accuracy = 100 * correct / total
            logger.debug(f"Accuracy of ONNX {model_name} is {accuracy:.2f}%")


if __name__ == "__main__":

    logger.debug("----------------- AI MODULE TEST: This is a DEBUG log --------------------")

    model_weights_path = PROJECT_ROOT / "ModelData" / "Weights"
    efficient_path = str(model_weights_path / "casting_efficientnet_b0.pth")
    
    efficient_info = {
        'module': 'torchvision.models',
        'model_name': "efficientnet_v2",
        'native': True,
        'distilled': False,
        'weights_path': efficient_path,
        'device': "cpu",
        'class_name': 'efficientnet_b0',
        'weights_class': 'EfficientNet_B0_Weights.DEFAULT',
        'image_size': 224,
        'num_classes': 2,
        'description': 'efficientnet_v2 from torchvision'
    }

    mobile_path = str(model_weights_path / "mobilenet_v2.pth")
    mobile_info = {
        'module': 'torchvision.models',
        'model_name': "mobilenet_v2",
        'native': True,
        'distilled': False,
        'weights_path': mobile_path,
        'device': "cpu",
        'class_name': 'mobilenet_v2',
        'weights_class': 'MobileNet_V2_Weights.DEFAULT',
        'image_size': 224,
        'num_classes': 2,
        'description': 'mobilenet_v2 from torchvision'

    }

    data_dir = str(PROJECT_ROOT / "ModelData" / "Dataset" / "casting_data")

    dataset_info = {
        'data_dir': data_dir,
        'batch_size': 32
    }

    efficientnet = AIModel(efficient_info)
    mobilenet = AIModel(mobile_info)
    dataset = DataWrapper()
    
    efficient_name = efficientnet.getInfo("model_name")
    mobile_name = mobilenet.getInfo("model_name")

    logger.debug(f"Inference {efficient_name} and {mobile_name}")
    logger.debug(f"Test inference on casting product")
    
    # First Inference test: with efficientnet
    dataset.loadInferenceData(model_info = efficient_info, dataset_info = dataset_info)
    inference_loader = dataset.getLoader()
    efficientnet.createOnnxModel(inference_loader)
    efficientnet.runInference(inference_loader)

    # Second Inference test: with mobilenet
    dataset.loadInferenceData(model_info = mobile_info, dataset_info = dataset_info)
    inference_loader = dataset.getLoader()
    mobilenet.createOnnxModel(inference_loader)
    mobilenet.runInference(inference_loader)

    logger.debug("------------- AI MODULE TEST END -------------------")


