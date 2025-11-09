""" Module for building, training, and saving the model. """

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import config
from data_loader import get_dataloaders

def get_model(num_classes=2):
    """Loads a pre-trained EfficientNet-B0 and replaces the classifier."""
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    # List all kind of layers of EfficientNet
    print("\n------ Layer Types of the model --------")
    unique_type_layers = set()
    for module in model.modules():
        unique_type_layers.add(module.__class__.__name__)

    for layer_type in sorted(list(unique_type_layers)):
        print(layer_type)
    

    # Freeze all the feature extractor layers
    for param in model.parameters():
        param.requires_grad = False

    # Get the number of input features
    num_ftrs = model.classifier[1].in_features
    
    # Replace the final layer with our new 2-class layer
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model

def train_model(model, dataloaders):
    """Trains the model and returns the trained model."""
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize the parameters of the new, unfrozen classifier
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=config.LEARNING_RATE)
    
    model = model.to(config.DEVICE)
    
    for epoch in range(config.NUM_EPOCHS):
        print(f'----- Epoch {epoch+1}/{config.NUM_EPOCHS} -----')
        print('-' * 10)

        # Set model to training mode
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        print(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f}")

    print("Training complete!")
    return model

def main():
    """Main function to run the training process."""
    print(f"Using device: {config.DEVICE}")
    
    # 1. Get Data
    dataloaders, class_names = get_dataloaders()
    
    # 2. Get Model
    model = get_model(num_classes=len(class_names))
    
    # 3. Train Model
    model = train_model(model, dataloaders)
    
    # 4. Save Model
    try:
        torch.save(model.state_dict(), config.MODEL_PATH)
        print(f"Model saved to {config.MODEL_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    main()