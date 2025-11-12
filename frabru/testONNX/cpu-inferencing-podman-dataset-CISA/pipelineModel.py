import os

#Local Scripts
from exportModel import load_model, exportModelONNX
from trainModel import trainModel
from modelConfig import ModelConfig
from inferModel import inferenceModel
from torch.utils.data import DataLoader



os.makedirs(ModelConfig.save_dir, exist_ok=True)

if __name__ == "__main__":

    model, test_dataset, test_loader = trainModel()
    print("TRAINING FINISHED. \n LOADING THE NEW MODEL...")
    #model_optimized = load_model()
    exportModelONNX(model, ModelConfig.model_name)

    inferenceModel(ModelConfig.model_name, test_dataset, test_loader, ModelConfig.device)


