from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

import onnx
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as tq
from abc import ABC, abstractmethod
from pathlib import Path
from copy import deepcopy
from torch.quantization import quantize_fx
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxruntime import quantization
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from BenchmarkingFactory.calibrationDataReader import CustomCalibrationDataReader
from BenchmarkingFactory.dataWrapper import DataWrapper
from BenchmarkingFactory.aiModel import AIModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Optimization(ABC): 

    @abstractmethod
    def applyOptimization(self):
        pass

    @abstractmethod
    def getOptimizationInfo(self, info: str):
        pass

class PruningOptimization(Optimization):

    def __init__(self, optimization_config):
        """
        Empty Constructor, load dynamically models to apply and create a copy of the new model
        """
        
        self.current_aimodel = None
        self.optimization_config = optimization_config

    def applyOptimization(self):
        """
        Apply the optimization pruning configured with config, to the aiModel attached

        Input:
            -N.A.

        Output:
            -optimized_model: aiModel optimized with global pruning
        """

        logger.debug(f"-----> [OPTIMIZATION MODULE] APPLY PRUNING OPTIMIZATION ")

        if not self.current_aimodel:
            raise MissingAIModelError(
                "No AIModel set. Call setAIModel() before applying optimization"
            )

        current_model_info = self.current_aimodel.getAllInfo()
        pruned_model_info = deepcopy(current_model_info)

        logger.debug(f"Creating a copy of the model {current_model_info['model_name']}")
        pruned_aimodel = AIModel(pruned_model_info)
        model_to_prune = pruned_aimodel.getModel()

        pruning_method = self.__getPruningMethod()
        layer_to_prune = self.__getLayerNames(model_to_prune)
        amount = self.getOptimizationInfo('amount')

        logger.info(f"Amount:{amount}")
        logger.info(f"Method:{pruning_method}")

        parameters_to_prune = []

        # Building list of parameters to prune
        for module in model_to_prune.modules():
            if module.__class__.__name__ in layer_to_prune:
                if hasattr(module, "weight"):
                    parameters_to_prune.append((module, "weight"))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method = pruning_method,
            amount = amount
        )

        # Applying pruning permanently
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        pruned_aimodel.getAllInfo()['model_name'] += "_pruned"
        pruned_aimodel.getAllInfo()['description'] += f"(Pruned with {self.getOptimizationInfo('method')} with amount {self.getOptimizationInfo('amount')})"
        logger.debug(f"<----- [OPTIMIZATION MODULE] APPLY PRUNING OPTIMIZATION")

        return pruned_aimodel

    def setOptimizationConfig(self, optimization_config):
        """
        Set the Optimization Config to apply

        Input:
            -optimization_config: config given from doe experiment
        """
        self.optimization_config = optimization_config

    def setAIModel(self, AIModel):
        """
        Set an AIModel in ordert to apply on it the pruning optimization
        """
        self.current_aimodel =  AIModel

    def getOptimizationInfo(self, info: str):
        """
        Return the choosen info from the optimization config setted
        """
        return self.optimization_config[info]

    def __getLayerNames(self, model: nn.Module):
        """
        Function to list Layers of model passed, in order to chose layers for global pruning
        
        Input:
            -model: torch module attached to current AIModule
        """

        if not self.current_aimodel:
            raise MissingAIModelError(
                "Internal error: __getLayerNames called but current_aimodel is not set."
            )
        
        model_name = self.current_aimodel.getInfo('model_name')
        unique_type_layers = set()

        for module in model.modules():
            unique_type_layers.add(module.__class__.__name__)

        logger.debug(f"\nGetting layers from model {model_name}")
        model_layers = sorted(list(unique_type_layers))
        for layer_type in model_layers:
            logger.debug(f"{layer_type}")

        # FOR THE FUTURE THESE LAYERS COULD BE PARAMETRIZED WITH CONFIG FILE (YAML/TOML)
        desired_global_pruning_layers = ["Conv2d", "Linear"]

        final_global_pruning_layers = {layer for layer in model_layers if layer in desired_global_pruning_layers}
        return final_global_pruning_layers

    def __getPruningMethod(self):
        """
        Function that get the right pruning method form prune.BasePruningMethods
        """

        pruning_method = self.getOptimizationInfo('method')
<<<<<<< HEAD
=======
        class_methods = ["RandomUnstructured", "L1Unstructured", "L2Unstructured"]
>>>>>>> main
        
        class_method = getattr(prune, pruning_method)

        if class_method is None:
            return prune.RandomUnstructured

        return class_method



class QuantizationOptimization(Optimization):

    def __init__(self, optimization_config):
        """
        Empty Constructor, load dynamically models to apply and create a copy of the new model
        """
        
        self.current_aimodel = None
        self.optimization_config = optimization_config

    def getOptimizationInfo(self, info: str):
        """
        Return the choosen info from the optimization config setted
        """
        return self.optimization_config[info]

    def setOptimizationConfig(self, optimization_config):
        """
        Set the Optimization Config to apply

        Input:
            -optimization_config: config given from doe experiment
        """
        self.optimization_config = optimization_config

    def setAIModel(self, AIModel):
        """
        Set an AIModel in ordert to apply on it the pruning optimization
        """
        self.current_aimodel =  AIModel

    def applyOptimization(self, input_examples):
        """
        Apply the optimization quantization configured with config, to the aiModel attached

        Input:
            -N.A.

        Output:
            -optimized_model: aiModel optimized with dynamic or static quantization
        """
        
        logger.debug(f"-----> [OPTIMIZATION MODULE] APPLY QUANTIZATION OPTIMIZATION")
    
        if not self.current_aimodel:
            raise MissingAIModelError(
                "No AIModel set. Call setAIModel() before applying optimization"
            )

        device = self.current_aimodel.getInfo('device')
        model_name = self.current_aimodel.getInfo('model_name')
        bit_method = self.getOptimizationInfo('method')
        quant_type = self.getOptimizationInfo('type')

        current_model_info = self.current_aimodel.getAllInfo()
        quantized_model_info = deepcopy(current_model_info)

        logger.debug(f"Creating a copy of the model {current_model_info['model_name']}")
        quantized_aimodel = AIModel(quantized_model_info)

        match quant_type:
            case "static":
                res = self.__staticQuantizationOnnx(model_name, device, input_examples)   
            
            case "dynamic":
                pass

        quantized_aimodel.getAllInfo()['model_name'] += "_quantized"
        quantized_aimodel.getAllInfo()['description'] += f"(Quantized with type:{self.getOptimizationInfo('type')} ; bit_method:{self.getOptimizationInfo('method')})"
        logger.debug(f"<----- [OPTIMIZATION MODULE] APPLY QUANTIZATION OPTIMIZATION")

        return quantized_aimodel

    def __staticQuantizationOnnx(self, model_name, device, inputs):
        """
        Apply Static Quantization with ONNX
        """  

        model_path = str(PROJECT_ROOT / "ModelData" / "ONNXModels" / f"{model_name}.onnx")
        model_prep_path = str(PROJECT_ROOT / "ModelData" / "ONNXModels" / f"{model_name}_prep.onnx")
        model_quantized_path = str(PROJECT_ROOT / "ModelData" / "ONNXModels" / f"{model_name}_quantized.onnx")

        onnx.shape_inference.infer_shapes_path(model_path, model_prep_path)

        quantization.shape_inference.quant_pre_process(
            model_prep_path, 
            model_prep_path, 
            skip_symbolic_shape=False
        )

        q_format = quantization.QuantFormat.QOperator

        q_static_opts = None

        if device == "gpu" or device == "cuda":
            q_static_opts = {
                "ActivationSymmetric": True,
                "WeighSymmetric": True
            }
        elif device == "cpu":
            q_static_opts = {
                "ActivationSymmetric": False,
                "WeightSymmetric": True
            }

        # Calibration dataset
        qdr = CustomCalibrationDataReader(inputs, input_name = "input")

        logger.info("Starting Static Quantization (QOperator)...")

        quantized_model = quantization.quantize_static(
            model_input=model_prep_path,
            model_output=model_quantized_path,
            calibration_data_reader=qdr,
            quant_format = q_format,
            per_channel=False,
            weight_type = QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            extra_options=q_static_opts
        )

        logger.info("Quantization Complete")

        if quantized_model is not None:
            return True
        else:
            return False
   
class MissingAIModelError(Exception):
    """
    Raised when an optimization is applied before an AIModel is set.
    """

    pass

if __name__ == "__main__":



    # Model Example
    model_weights_path = PROJECT_ROOT / "ModelData" / "Weights"

    # Dataset Example
    data_dir = str(PROJECT_ROOT / "ModelData" / "Dataset" / "casting_data")
    dataset_info = {
        'data_dir': data_dir,
        'batch_size': 32
    }
    dataset = DataWrapper()



    # -------- TEST WITH BASE MODEL ---------------

    efficient_path = str(model_weights_path / "casting_efficientnet_b0.pth")

    # Faking user input
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

    # Creating base model
    efficientnet = AIModel(efficient_info)
    logger.debug(f"Inference with {efficientnet.getInfo('model_name')}, description: {efficientnet.getInfo('description')}")
    logger.debug(f"Test inference on casting product")

    # Attaching dataset to unpruned model
    dataset.loadInferenceData(model_info = efficientnet.getAllInfo(), dataset_info = dataset_info)
    inference_loader = dataset.getLoader()

    # Inference with unpruned model
    efficientnet.createOnnxModel(inference_loader)
    efficientnet.runInference(inference_loader)

    # Base model for mobilenet
    mobilenet = AIModel(mobile_info)
    mobilenet_dataset = DataWrapper()
    mobilenet_dataset.loadInferenceData(model_info = mobile_info, dataset_info = dataset_info)
    mobilenet_loader = dataset.getLoader()
    mobilenet.createOnnxModel(mobilenet_loader)
    mobilenet.runInference(mobilenet_loader)

    # -----------------------------------------------

    # -------- TEST WITH PRUNED MODEL -------------------
    #Pruning Info example
    pruning_info = {
        "method": "L1Unstructured",
        "amount": 0.3
    }

    # Creating Pruning Optimization Object
    pruning_optimizator = PruningOptimization(pruning_info)
    
    # Creating new Pruned AIModel
    pruning_optimizator.setAIModel(efficientnet)
    pruned_efficientnet = pruning_optimizator.applyOptimization()

    # Attaching dataset to pruned model
    dataset.loadInferenceData(model_info = pruned_efficientnet.getAllInfo(), dataset_info = dataset_info)
    inference_loader = dataset.getLoader()

    # Inference with prunde model
    pruned_efficientnet.createOnnxModel(inference_loader)
    pruned_efficientnet.runInference(inference_loader)

    # -----------------------------------------------

    # ---------- TEST WITH QUANTIZED MODEL ---------------

    quantization_info = {
        "arch": "x86", #aarch
        "method": "QInt8",
        "type": "static"
    }

    quantization_optimizator = QuantizationOptimization(quantization_info)
    
    # Creating new quantized AIModel
    quantization_optimizator.setAIModel(efficientnet)
    quantized_efficientnet = quantization_optimizator.applyOptimization(inference_loader)

    # Attaching dataset to quantized model
    dataset.loadInferenceData(model_info = quantized_efficientnet.getAllInfo(), dataset_info = dataset_info)
    inference_loader = dataset.getLoader()

    quantized_efficientnet.runInference(inference_loader)

    quantization_optimizator.setAIModel(mobilenet)
    quantized_mobilenet = quantization_optimizator.applyOptimization(mobilenet_loader)
    mobilenet_dataset.loadInferenceData(model_info = quantized_mobilenet.getAllInfo(), dataset_info = dataset_info)
    mobilenet_loader = mobilenet_dataset.getLoader()
    quantized_mobilenet.runInference(mobilenet_loader)


    # -----------------------------------------------



