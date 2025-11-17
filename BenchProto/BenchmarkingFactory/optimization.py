from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

import torch.nn as nn
import torch.nn.utils.prune as prune
from abc import ABC, abstractmethod
from pathlib import Path
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

        if not self.current_aimodel:
            raise MissingAIModelError(
                "No AIModel set. Call setAIModel() before applying optimization"
            )

        current_model_info = self.current_aimodel.model_info

        logger.debug(f"Creating a copy of the model {current_model_info['model_name']}")
        pruned_aimodel = AIModel(current_model_info)
        model_to_prune = pruned_aimodel.getModel()

        pruning_method = self.__getPruningMethod()
        layer_to_prune = self.__getLayerNames(model_to_prune)
        amount = self.getOptimizationInfo('amount')

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

        pruned_aimodel.model_info['model_name'] += "_pruned"
        pruned_aimodel.model_info['description'] += f"(Pruned with {self.getOptimizationInfo('Pruning')} with amount {self.getOptimizationInfo('amount')})"
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

        pruning_method = self.getOptimizationInfo('Pruning')
        class_methods = ["RandomUnstructured", "L1Unstructured", "L2Unstructured"]
        
        if pruning_method not in class_methods:
            logger.error(f"Pruning method {pruning_method} not supported, fall back on RandomPruning")
        class_method = getattr(prune, pruning_method)

        if class_method is None:
            return prune.RandomUnstructured

        return class_method



class QuantizationOptimization(Optimization):

    def __init__(self):
        pass

    def applyOptimization(self):
        pass


class MissingAIModelError(Exception):
    """
    Raised when an optimization is applied before an AIModel is set.
    """

    pass

if __name__ == "__main__":


    # Model Example
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

    # Dataset Example
    data_dir = str(PROJECT_ROOT / "ModelData" / "Dataset" / "casting_data")
    dataset_info = {
        'data_dir': data_dir,
        'batch_size': 32
    }

    # Pruning Info example
    pruning_info = {
        "Pruning": "RandomUnstructured",
        "amount": 0.3
    }

    efficientnet = AIModel(efficient_info)
    dataset = DataWrapper()
    pruning_optimizator = PruningOptimization(pruning_info)
    
    logger.debug(f"Inference with {efficientnet.getInfo('model_name')}, description: {efficientnet.getInfo('description')}")
    logger.debug(f"Test inference on casting product")

    # Creating new Pruned AIModel
    pruning_optimizator.setAIModel(efficientnet)
    pruned_efficientnet = pruning_optimizator.applyOptimization()

    # Attaching dataset to pruned model
    dataset.loadInferenceData(model_info = pruned_efficientnet.model_info, dataset_info = dataset_info)
    inference_loader = dataset.getLoader()

    # Inference with prunde model
    pruned_efficientnet.createOnnxModel(inference_loader)
    pruned_efficientnet.runInference(inference_loader)



