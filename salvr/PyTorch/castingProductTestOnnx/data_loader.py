""" Data loadnin and preprocessing """

import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import config 

weights = models.EfficientNet_B0_Weights.DEFAULT

def get_model_transforms():

    """
    Returns the data transforms for training and testing.
    """

    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE + 32), # 256
            weights.transforms() # Apply the model's specific transforms
        ]),
        "test": transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE + 32), # 256
            weights.transforms() # Apply the model's specific transforms
        ]),
    }

    return data_transforms

def get_dataloaders(): 

    """
    Creates and returns the train and test dataloaders
    """

    data_transforms = get_model_transforms()

    print(f"Loading data from {config.DATA_DIR}")

    # Image dataset loading, a dictionary of two ImageFolder
    image_dataset = {
        "train": datasets.ImageFolder(
            f"{config.DATA_DIR}/train",
            data_transforms['train']
        ),
        "test": datasets.ImageFolder(
            f"{config.DATA_DIR}/test",
            data_transforms['test']
        )
    }

    # Dataloaders
    dataloaders = {
        "train": DataLoader(
            image_dataset['train'],
            batch_size = config.BATCH_SIZE,
            shuffle = True
        ),
        "test": DataLoader(
            image_dataset['test'],
            batch_size = config.BATCH_SIZE,
            shuffle = True
        )
    }

    class_names = image_dataset['train'].classes
    print(f"Classes found: {class_names}")

    return dataloaders, class_names

if __name__ == "__main__":
    print("Testing data loader...")
    loaders, names = get_dataloaders()
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Test batches: {len(loaders['test'])}")


