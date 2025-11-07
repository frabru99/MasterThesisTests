import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm


#Local Scripts
from datasetPCB import PCBDataset, load_data, train_transform, val_transform, test_transform
from modelConfig import ModelConfig, modelCreation



def train_epoch(model, dataloader, criterion, optimizer, device):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/len(dataloader):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc



def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total



def dataPrep():

    # Load data
    data_list_training, data_list_test = load_data(fraction=ModelConfig.train_perc)
    
    if len(data_list_training) ==0 or len(data_list_test) == 0:
        print("No data found! Please check your directory structure and CSV files.")
        return
    
    # Split data
    train_data, val_data = train_test_split(
        data_list_training, 
        test_size=ModelConfig.val_perc, 
        random_state=42,
        stratify=[label for _, label in data_list_training]
    )

    
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    # Create datasets and dataloaders
    train_dataset = PCBDataset(train_data, transform=train_transform)
    val_dataset = PCBDataset(val_data, transform=val_transform)
    test_dataset = PCBDataset(data_list_test, transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=ModelConfig.batch_size,
        shuffle=True,
        num_workers=ModelConfig.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=ModelConfig.batch_size,
        shuffle=False,
        num_workers=ModelConfig.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        pin_memory=True,
        batch_size=ModelConfig.batch_size,
        shuffle=False,
        num_workers=ModelConfig.num_workers
    )

    return train_dataset, val_dataset, train_loader, val_loader, test_dataset, test_loader


def trainModel():
    print(f"Using device: {ModelConfig.device}")
    
    train_dataset, val_dataset, train_loader, val_loader, test_dataset, test_loader = dataPrep()
    model, criterion, optimizer, scheduler = modelCreation()
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\nStarting training...")
    for epoch in range(ModelConfig.num_epochs):
        print(f"\nEpoch {epoch+1}/{ModelConfig.num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, ModelConfig.device)
        val_loss, val_acc = validate(model, val_loader, criterion, ModelConfig.device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': ModelConfig.classes
            }, os.path.join(ModelConfig.save_dir, 'best_model.pth'))
            print(f"✓ Saved best model with val_acc: {val_acc:.2f}%")
    
    # Save final model
    torch.save({
        'epoch': ModelConfig.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'classes': ModelConfig.classes
    }, os.path.join(ModelConfig.save_dir, 'final_model.pth'))
    
    # Save training history
    with open(os.path.join(ModelConfig.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved in '{ModelConfig.save_dir}' directory")

    return model, test_dataset, test_loader