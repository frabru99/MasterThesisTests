from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.optim as optim


class ModelConfig:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)                                                        
    model_name = "mobilenet_v2"
    data_root = "./data" #We should have an abstraction method to access this folder
    pcb_dirs = ["pcb1", "pcb2", "pcb3", "pcb4"]
    classes = ["normal", "melt", "scratch", "bent", "missing","burnt", "dirt", "damage", "wrong place"]
    img_size= 224
    batch_size= 15
    num_epochs = 3
    learning_rate = 0.0001
    train_perc = 0.7
    val_perc=0.2
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "models"

def modelCreation():

    # Create model
    model = ModelConfig.model

    in_features = model.classifier[1].in_features

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(in_features, len(ModelConfig.classes)))

    model = model.to(ModelConfig.device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=ModelConfig.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    return model, criterion, optimizer, scheduler
