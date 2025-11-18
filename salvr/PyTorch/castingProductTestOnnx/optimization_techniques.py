import torch
import torch.nn as nn
import onnx
import torch.nn.utils.prune as prune
import torch.nn.functional as F

def global_pruning(model: nn.Module, layer_type_names: list, pruning_method: prune.BasePruningMethod, amount: float=0.2):

    """
    Prunes a model globally by removing 'amount' 
    of the lowest magnitude weights across type_layers passed
    """

    torch.manual_seed(42)

    parameters_to_prune = []


    # Checking what module should be pruned in the model
    for module in model.modules():
        if module.__class__.__name__ in layer_type_names:
            if hasattr(module, 'weight'):
                parameters_to_prune.append((module, 'weight'))

    # Applying pruning to the chosen modules
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=pruning_method,
        amount = amount,
    )

    # Applying pruning permanently
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
        