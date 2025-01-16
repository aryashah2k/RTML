# For our puffer surver we need to browse via a proxy!!
import os
# Set HTTP and HTTPS proxy
os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import AlexNet_Weights, GoogLeNet_Weights
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import copy
from torch.utils.data import DataLoader
from torchsummary import torchsummary
from alexnet import AlexNet
from googlenet import GoogLeNet
import torchvision.models as models

def count_parameters(model):
    """Count number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, dataloader, criterion, optimizer, device, is_inception=False):
    """Train one epoch and return average loss and accuracy"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_time = 0
    batches = 0
    
    for inputs, labels in dataloader:
        batch_start = time.time()
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            if is_inception:
                outputs = model(inputs)
                # Handle different output formats for custom vs pretrained GoogLeNet
                if isinstance(outputs, tuple):
                    if len(outputs) == 3:  # Custom GoogLeNet (output, aux1, aux2)
                        output, aux1, aux2 = outputs
                        loss1 = criterion(output, labels)
                        loss2 = criterion(aux1, labels)
                        loss3 = criterion(aux2, labels)
                        loss = loss1 + 0.3 * loss2 + 0.3 * loss3
                        outputs = output  # Use main output for accuracy
                    else:  # Pretrained GoogLeNet (output, aux_outputs)
                        output, aux_outputs = outputs
                        loss1 = criterion(output, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.3 * loss2
                        outputs = output  # Use main output for accuracy
                else:
                    outputs = outputs
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
        
        batch_time = time.time() - batch_start
        total_time += batch_time
        batches += 1
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    avg_batch_time = total_time / batches
    
    return epoch_loss, epoch_acc.item(), avg_batch_time

def evaluate(model, dataloader, criterion, device, is_inception=False):
    """Evaluate model on dataloader"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if is_inception:
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take only the main output
            else:
                outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc.item()

def evaluate_test(model, test_loader, criterion, device, is_inception=False):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if is_inception:
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            else:
                outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc

def plot_training_comparison(histories, title):
    """Plot training histories for multiple models"""
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    for model_name, history in histories.items():
        plt.plot(history['train_loss'], label=f'{model_name} (train)')
        plt.plot(history['val_loss'], label=f'{model_name} (val)')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    for model_name, history in histories.items():
        plt.plot(history['train_acc'], label=f'{model_name} (train)')
        plt.plot(history['val_acc'], label=f'{model_name} (val)')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

def modify_pretrained_alexnet(model, num_classes=10):
    """Modify pretrained AlexNet for CIFAR-10"""
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def modify_pretrained_googlenet(model, num_classes=10):
    """Modify pretrained GoogLeNet for CIFAR-10"""
    model.fc = nn.Linear(1024, num_classes)
    return model

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=20, is_inception=False):
    """Train model and return model, history, and best accuracy"""
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'batch_times': []
    }
    
    best_acc = 0.0
    training_start = time.time()
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc, batch_time = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device, 
            is_inception=is_inception
        )
        history['batch_times'].append(batch_time)
        
        # Evaluate
        val_loss, val_acc = evaluate(
            model, dataloaders['val'], criterion, device,
            is_inception=is_inception
        )
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'{type(model).__name__.lower()}_best.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Batch Time: {batch_time:.4f}s')
    
    return model, history, best_acc

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    # For custom models (CIFAR-10 is 32x32)
    transform_custom = transforms.Compose([
        transforms.Resize(224),  # Resize to match pretrained models
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_custom_train = transforms.Compose([
        transforms.Resize(224),  # Resize to match pretrained models
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # For pretrained models (using ImageNet normalization)
    transform_pretrained = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_pretrained_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets with appropriate transforms
    def get_datasets(transform_train, transform_test):
        full_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        train_size = 45000  # 90% of training data
        val_size = 5000    # 10% of training data
        trainset, valset = torch.utils.data.random_split(full_trainset, [train_size, val_size])
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        return trainset, valset, testset
    
    # Get datasets for both custom and pretrained models
    trainset_custom, valset_custom, testset_custom = get_datasets(transform_custom_train, transform_custom)
    trainset_pretrained, valset_pretrained, testset_pretrained = get_datasets(transform_pretrained_train, transform_pretrained)
    
    # Create dataloaders
    batch_size = 128
    def get_dataloaders(trainset, valset, testset):
        return {
            'train': DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2),
            'val': DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2),
            'test': DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        }
    
    dataloaders_custom = get_dataloaders(trainset_custom, valset_custom, testset_custom)
    dataloaders_pretrained = get_dataloaders(trainset_pretrained, valset_pretrained, testset_pretrained)
    
    # Initialize all models
    models_to_train = {
        # Custom models with different configurations
        'AlexNet with LRN': AlexNet(num_classes=10, use_lrn=True),
        'AlexNet without LRN': AlexNet(num_classes=10, use_lrn=False),
        'Custom GoogLeNet': GoogLeNet(num_classes=10),
        # Pretrained models
        'Pretrained AlexNet': modify_pretrained_alexnet(models.alexnet(weights=AlexNet_Weights.DEFAULT), num_classes=10),
        'Pretrained GoogLeNet': modify_pretrained_googlenet(models.googlenet(weights=GoogLeNet_Weights.DEFAULT), num_classes=10)
    }
    
    # Compare number of parameters
    param_counts = {}
    for name, model in models_to_train.items():
        models_to_train[name] = model.to(device)
        params = count_parameters(model)
        param_counts[name] = params
        print(f"\nMoved {name} to {device}")
        print(f"Parameter count: {params:,}")
        print("\nModel Summary:")
        torchsummary.summary(model, (3, 224, 224))
    
    # Initialize tracking variables
    criterion = nn.CrossEntropyLoss()
    histories = {}
    best_accuracies = {}
    training_times = {}
    
    # Different learning rates for custom and pretrained models
    lr_custom = 0.01
    lr_pretrained = 0.001
    
    # Train all models
    for name, model in models_to_train.items():
        print(f"\nTraining {name}...")
        
        # Select appropriate learning rate and dataloaders
        is_pretrained = 'Pretrained' in name
        lr = lr_pretrained if is_pretrained else lr_custom
        dataloaders = dataloaders_pretrained if is_pretrained else dataloaders_custom
        
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        
        # Record training start time
        training_start = time.time()
        
        # Train model
        model, history, best_acc = train_model(
            model, dataloaders, criterion, optimizer, device,
            num_epochs=20, is_inception=('GoogLeNet' in name)
        )
        
        # Record total training time
        training_times[name] = time.time() - training_start
        histories[name] = history
        best_accuracies[name] = best_acc
        
        print(f"Total training time for {name}: {training_times[name]/60:.2f} minutes")
    
    # Evaluate models on test set
    print("\nEvaluating models on test set:")
    test_results = {}
    
    for name, model in models_to_train.items():
        is_pretrained = 'Pretrained' in name
        dataloaders = dataloaders_pretrained if is_pretrained else dataloaders_custom
        
        test_loss, test_acc = evaluate_test(
            model, dataloaders['test'], criterion, device,
            is_inception=('GoogLeNet' in name)
        )
        test_results[name] = {
            'loss': test_loss,
            'accuracy': test_acc
        }
        print(f"\n{name}:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Plot training comparison
    plot_training_comparison(histories, 'Model Comparison on CIFAR-10')
    
    # Create comparison table
    comparison_data = []
    for name in models_to_train.keys():
        final_train_loss = histories[name]['train_loss'][-1]
        final_train_acc = histories[name]['train_acc'][-1]
        final_val_loss = histories[name]['val_loss'][-1]
        final_val_acc = histories[name]['val_acc'][-1]
        test_loss = test_results[name]['loss']
        test_acc = test_results[name]['accuracy']
        
        comparison_data.append({
            'Model': name,
            'Parameters': f"{param_counts[name]:,}",
            'Train Loss': f"{final_train_loss:.4f}",
            'Train Acc': f"{final_train_acc:.2%}",
            'Val Loss': f"{final_val_loss:.4f}",
            'Val Acc': f"{final_val_acc:.2%}",
            'Test Loss': f"{test_loss:.4f}",
            'Test Acc': f"{test_acc:.2%}",
            'Best Val Acc': f"{best_accuracies[name]:.2%}",
            'Training Time': f"{training_times[name]/60:.1f} min",
            'Avg Batch Time': f"{np.mean(histories[name]['batch_times'])*1000:.1f} ms"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nFinal Model Comparison Summary:")
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)        # Don't wrap wide tables
    print(comparison_df.to_string(index=False))
    
    # Save results
    comparison_df.to_csv('model_comparison.csv', index=False)
    print("\nResults saved to 'model_comparison.csv' and 'model_comparison.png'")

if __name__ == '__main__':
    main()