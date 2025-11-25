
from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

import numpy as np
import onnxruntime as ort
import onnx
import torch
import json
from os import remove, mkdir
from importlib import import_module
from pathlib import Path
from torchvision import models
from BenchmarkingFactory.dataWrapper import DataWrapper
from pymemtrace import cPyMemTrace
import torch.nn as nn

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

        self.__model_info = model_info
        self.__model = self._loadModel(model_info['module'], model_info['class_name'])

    def getAllInfo(self):
        return self.__model_info

    def getInfo(self, info_name: str):

        return self.__model_info[info_name]

    def __printStats(self, input: dict, topic: str) -> None:
        """
        Handler function to print Stas of the model on terminal.

        Input:
            - input: dict that contains couples key, value to print.
            - topic: the topic to print at the first line

        """

        print("\n" +"-"*10 + topic + "-"*10)
        for key, value in input.items():
            if key=="Accuracy":
                print("\n")
                
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
                continue
            
            print(f"{key}: {value}")
                

        print("-"*10 + "-"*len(topic)+"-"*10+"\n")
        


    def _replaceModelClassifier(self, model, class_name):
        """
        Function that automatically recognize the final layer and replace for the inference on a specific dataset

        Inputs:
            - model: Model passed that needs to be recognized
        """

        logger.debug(f"-----> [AIMODEL MODULE] REPLACE MODEL CLASSIFIER")

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
        logger.debug(f"<----- [AIMODEL MODULE] REPLACE MODEL CLASSIFIER")

    def _loadModel(self, module, class_name): 
        """
        Function to reconstruct the model from module and the class name

        Input: 
            - module: Module from PyTorch Library (e.g., torchvision.models)
            - class_name: Attribute name of the module, to load the model structure
        """
        
        logger.debug(f"-----> [AIMODEL MODULE] LOAD MODEL")

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

            logger.info(f"MODEL LOADED CORRECTLY: {class_name}")

        except FileNotFoundError:
            logger.error(f"Weight file not found at {self.getInfo('weights_path')}. Model has random weights.")
        except Exception as e:
            logger.error(f"Error loading model weights from {self.getInfo('weights_path')}: {e}")

        logger.debug(f"<----- [AIMODEL MODULE] LOAD MODEL\n")

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


    def __calculateStats(self, profile_file_path: str, num_batches: int, total_images: int, correct: int, total: int, running_loss: int) -> dict:
        """
        Parses an ONNX Runtime profile JSON file to get pure kernel statistics.
        
        Input:
            -profile_file_path: The path to the profile.json file.
            -num_batches: The total number of batches in the inference run
                        (e.g., len(inference_loader)).
            -total_images: The total number of images in the dataset
                        (e.g., len(inference_loader.dataset)).

        Output:
            A dictionary with total time, per-batch avg, and per-image avg.
        """

        total_kernel_time_us = 0
        total_model_run_time_us = 0
        total_sequential_execution_time_us = 0
        node_events = []
        #model_run_events = []
        #linear_execution_events = []

        try:
            with open(profile_file_path, 'r') as f:
                trace_data = json.load(f)

            # Iterate over all events in the trace
            for event in trace_data:
                event_cat = event.get("cat")
                event_name = event.get("name")
                event_dur = event.get("dur", 0)

                if event_cat  == "Node":
                    duration_us = event_dur
                    total_kernel_time_us += duration_us
                    node_events.append(duration_us)
                elif event_cat == "Session" and event_name == "model_run":
                    model_run_duration_us = event_dur
                    total_model_run_time_us += model_run_duration_us
                    #model_run_events.append(model_run_duration_us)
                elif event_cat == "Session" and event_name == "SequentialExecutor::Execute":
                    linear_execution_duration_us = event_dur
                    total_sequential_execution_time_us += linear_execution_duration_us
                    #linear_execution_events.append(linear_execution_duration_us)
            
            if num_batches == 0 or total_images == 0:
                logger.error(f"Number of batches or images cannot be zero. im: {total_images}; batch: {num_batches}")
                return {}

            if not node_events:
                logger.warning(f"No Node events found in {profile_file_path}.")
                return {}

            # Calculate the stats to return
            total_kernel_time_ms = total_kernel_time_us / 1000.0
            total_model_run_time_ms = total_model_run_time_us / 1000.0
            total_sequential_execution_time_ms = total_sequential_execution_time_us / 1000.0

            avg_kernel_time_per_batch_ms = total_kernel_time_ms / num_batches
            avg_kernel_time_per_image_ms = total_kernel_time_ms / total_images
            avg_sequential_executor_time_per_batch_ms = total_sequential_execution_time_ms / num_batches
            avg_sequential_executor_time_per_image_ms = total_sequential_execution_time_ms / total_images
            avg_model_run_time_per_batch_ms = total_model_run_time_ms / num_batches
            avg_model_run_time_per_image_ms = total_model_run_time_ms / total_images
            
            total_onnx_runtime_overhead = total_model_run_time_ms - total_sequential_execution_time_ms
            avg_onnx_runtime_overhead_per_batch_ms = avg_model_run_time_per_batch_ms - avg_sequential_executor_time_per_batch_ms
            avg_onnx_runtime_overhead_per_image_ms = avg_model_run_time_per_image_ms - avg_sequential_executor_time_per_image_ms
            

            node_latencies_ms = [n / 1000.0 for n in node_events]
            p95_node_latency_ms = np.percentile(node_latencies_ms, 95)

            accuracy = 100 * correct / total
            average_loss = running_loss / total

            stats = {
                "Inference Event Path": profile_file_path,
                "Total 'kernel' inference time": total_kernel_time_ms,
                "Avg. 'kernel' inference time per batch": avg_kernel_time_per_batch_ms,
                "Avg. 'kernel' inference time per image": avg_kernel_time_per_image_ms,
                "Total sequential executor time": total_sequential_execution_time_ms,
                "Avg. sequential executor time per batch": avg_sequential_executor_time_per_batch_ms,
                "Avg. sequential executor time per image": avg_sequential_executor_time_per_image_ms,
                "Total model run time": total_model_run_time_ms,
                "Avg. model run time per batch": avg_model_run_time_per_batch_ms,
                "Avg. model run time per image": avg_model_run_time_per_image_ms,
                "Total ONNX runtime overhead": total_onnx_runtime_overhead,
                "Avg. ONNX runtime overhead per batch": avg_onnx_runtime_overhead_per_batch_ms,
                "Avg. ONNX runtime overhead per image": avg_onnx_runtime_overhead_per_batch_ms,
                "total_nodes_executed": len(node_events),
                "p95_node_latency_ms": p95_node_latency_ms,
                "Accuracy": accuracy,
                "Avg. Loss": average_loss
            }

        
            # Clean up the file
            try:
                remove(profile_file_path)
                logger.debug(f"Cleaned up profile file: {profile_file_path}")
            except OSError as e:
                logger.warning(f"Could not delete profile file {profile_file_path}: {e}")


            return stats

        except FileNotFoundError:
            logger.error(f"Profile file not found: {profile_file_path}")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {profile_file_path}")
            return {}
        except Exception as e:
            logger.error(f"An error occured during profiling: {e}")
            return {}

    def setModel(self, model):
        """
        Set new model to the AIModel

        Input:
            - model: new Pytorch nn module 
        """

        self.__model = model

    def getModel(self):
        """
        Returns the torch model attached to aimodel
        """

        return self.__model

    def createOnnxModel(self, input_data, config_id):
        """
        Function that creates the onnx file to use for the inference part
        """

        logger.debug(f"-----> [AIMODEL MODULE] CREATE ONNX MODEL")


        model_name = self.getInfo('model_name')
        onnx_directory_path = PROJECT_ROOT / "ModelData" / "ONNXModels" / f"{config_id}" 
        onnx_model_path = onnx_directory_path /f"{model_name}.onnx"


        if not onnx_directory_path.exists():
            mkdir(onnx_directory_path)

        
        if onnx_model_path.exists():
            logger.info(f"ONNX file of {model_name} already exists at {onnx_model_path}")
            #return TO PASS THE CREATION IF IT ALREADY EXISTS 

        # Getting parameters
        onnx_model_path = str(onnx_model_path)
        device = self.getInfo('device')
        image_size = self.getInfo('image_size')

        self.getModel().to(device)
        self.getModel().eval()

        dummy_input = None

        try:
            inputs, _ = next(iter(input_data))

            dummy_input = inputs[0:1].to(device)
            logger.debug("Generated dummy input from real dataset (sliced to batch size 1)")

        except (StopIteration, TypeError, AttributeError):
            logger.warning("Could not fetch data from loader. Generating random dummy input.")
            dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

        # Dynamic axes config (Standard exporter)
        dynamic_axes_config = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }

        try:
            torch.onnx.export(
                self.getModel(),
                dummy_input,
                onnx_model_path,
                export_params=True,
                opset_version=13,
                do_constant_folding = True,
                input_names = ['input'],
                output_names = ['output'],
                dynamic_axes = dynamic_axes_config,
                dynamo = False
            )
            logger.info(f"ONNX Model successfully created at {onnx_model_path}")
        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")

        logger.debug(f"<----- [AIMODEL MODULE] CREATE ONNX MODEL\n")

    def runInference(self, input_data, config_id) -> str:
        """
        Function to run inference on a given dataset

        Inputs: 
            - input_data: Dataset passed from the dataloader

        Outputs:
            - stats_path: Path of 
        """
        
        logger.debug(f"-----> [AIMODEL MODULE] RUN INFERENCE")

        device_str = self.getInfo('device')
        model_name = self.getInfo('model_name')
        onnx_model_path = PROJECT_ROOT / "ModelData" / "ONNXModels"  / f"{config_id}" / f"{model_name}.onnx"

        provider_list = self._getProviderList(self.getInfo('device'))
        device_name = "cuda" if device_str == "gpu" else "cpu"

        try:
            # Enable profiling
            sess_options = ort.SessionOptions()
            sess_options.enable_mem_pattern = True
            sess_options.enable_profiling = True
            sess_options.profile_file_prefix = self.getInfo('model_name')
            logger.debug(f"Session is enabled with profiling")

            ort_session = ort.InferenceSession(str(onnx_model_path), providers=provider_list, sess_options = sess_options)
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            
            input_type = np.float32
            output_type = np.float32

            logger.debug(f"Input of ort session: {ort_session.get_inputs()[0]}")
            logger.debug(f"Output of ort session: {ort_session.get_outputs()[0]}")

        except Exception as e:
            logger.debug(f"Error loading ONNX model: {e}")

        io_binding = ort_session.io_binding()

        n_total_images = len(input_data.dataset)
        num_batches = len(input_data)
        logger.info(f"In this dataset there are {n_total_images} images across {num_batches} batches")
        
        total = 0
        correct = 0
        running_loss = 0
        criterion = nn.CrossEntropyLoss()

    
        with torch.no_grad():
            with cPyMemTrace.Profile(0):
                for inputs, labels in input_data:

                    labels = labels.to(device_str)
                    batch_size = inputs.shape[0]

                    if device_name == "cuda":

                        # Moving the input on the same device of the onnx session for zero copy
                        input_tensor = inputs.to(device_str).contiguous()

                        # Pre allocate output tensor on gpu
                        output_shape = (batch_size, output_num_classes)
                        onnx_output_tensor = torch.empty(output_shape, 
                                                        dtype=torch.float32, 
                                                        device=device_name).contiguous()

                        # Binding Input and Outputs
                        io_binding.bind_input(
                            name=input_name,
                            device_type='cuda',
                            device_id=0,
                            element_type=input_type,
                            shape=tuple(input_tensor.shape),
                            buffer_ptr=input_tensor.data_ptr()
                        )
                        io_binding.bind_output(
                            name=output_name,
                            device_type='cuda',
                            device_id=0,
                            element_type=output_type,
                            shape=tuple(onnx_output_tensor.shape),
                            buffer_ptr=onnx_output_tensor.data_ptr()
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

                    loss = criterion(onnx_outputs_tensor, labels)
                    running_loss += loss.item() * batch_size
                    predicted_indices = torch.argmax(onnx_outputs_tensor, dim=1)
                    total += labels.shape[0]
                    correct += (predicted_indices == labels).sum().item()

            # Get profile path
            profile_file_path = ort_session.end_profiling()
            if not profile_file_path:
                logger.error(f"Profiling enabled but no file was generated.")
                return {}

            logger.debug(f"Profile file generated: {profile_file_path}")
            
            # Get kernel stats
            stats = self.__calculateStats(profile_file_path, num_batches, n_total_images, correct, total, running_loss)

            
            self.__printStats(stats, f" {model_name} STATS ")            

            logger.debug(f"<----- [AIMODEL MODULE] RUN INFERENCE\n")

            return stats

    

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

    data_dir = "ModelData/Dataset/casting_data"

    dataset_info = {
        'data_dir': data_dir,
        'batch_size': 32
    }

    efficientnet = AIModel(efficient_info)
    mobilenet = AIModel(mobile_info)
    dataset = DataWrapper()
    
    efficient_name = efficientnet.getInfo("model_name")
    mobile_name = mobilenet.getInfo("model_name")
    
    # First Inference test: with efficientnet
    dataset.loadInferenceData(model_info = efficient_info, dataset_info = dataset_info)
    inference_loader = dataset.getLoader()
    efficientnet.createOnnxModel(inference_loader)
    efficient_stats = efficientnet.runInference(inference_loader)

    # Second Inference test: with mobilenet
    dataset.loadInferenceData(model_info = mobile_info, dataset_info = dataset_info)
    inference_loader = dataset.getLoader()
    mobilenet.createOnnxModel(inference_loader)
    mobile_stats = mobilenet.runInference(inference_loader)

    logger.debug("------------- AI MODULE TEST END -------------------")

