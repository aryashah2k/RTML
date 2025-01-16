# For our puffer surver we need to browse via a proxy!!
import os
# Set HTTP and HTTPS proxy
os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt
import numpy as np

class AlexNetGPU1(nn.Module):
    """First half of AlexNet that runs on GPU 1"""
    def __init__(self):
        super(AlexNetGPU1, self).__init__()
        self.features = nn.Sequential(
            # First convolutional layer (on GPU 1)
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # 48 filters
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Second convolutional layer (on GPU 1)
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Third convolutional layer (on GPU 1)
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fourth convolutional layer (on GPU 1)
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fifth convolutional layer (on GPU 1)
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
    def forward(self, x):
        x = self.features(x)
        return x

class AlexNetGPU2(nn.Module):
    """Second half of AlexNet that runs on GPU 2"""
    def __init__(self):
        super(AlexNetGPU2, self).__init__()
        self.features = nn.Sequential(
            # First convolutional layer (on GPU 2)
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # 48 filters
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Second convolutional layer (on GPU 2)
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Third convolutional layer (on GPU 2)
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fourth convolutional layer (on GPU 2)
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fifth convolutional layer (on GPU 2)
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
    def forward(self, x):
        x = self.features(x)
        return x

class AlexNetMultiGPU(nn.Module):
    """Complete AlexNet split across two GPUs"""
    def __init__(self, num_classes=10):
        super(AlexNetMultiGPU, self).__init__()
        self.gpu1_stream = torch.cuda.Stream(device='cuda:0')
        self.gpu2_stream = torch.cuda.Stream(device='cuda:1')
        
        self.gpu1_net = AlexNetGPU1().cuda(0)
        self.gpu2_net = AlexNetGPU2().cuda(1)
        
        # Classifier runs on GPU 1
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),  # 256 = 128 + 128 channels from both GPUs
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        ).cuda(0)
        
    def forward(self, x):
        # Input is already on GPU 0
        batch_size = x.size(0)
        
        # Process on GPU 1
        with torch.cuda.stream(self.gpu1_stream):
            output1 = self.gpu1_net(x)  # Use full input
            
        # Move input to GPU 2 and process
        with torch.cuda.stream(self.gpu2_stream):
            x2 = x.cuda(1)  # Move to GPU 2
            output2 = self.gpu2_net(x2)  # Process full input
            
        # Synchronize the streams
        torch.cuda.synchronize()
        
        # Move output2 to GPU 1 and concatenate
        output2 = output2.cuda(0)
        output = torch.cat([output1, output2], dim=1)  # Concatenate along channel dimension
        
        # Flatten and pass through classifier
        output = output.view(batch_size, -1)
        output = self.classifier(output)
        
        return output

def train_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Move input and target to GPU 0 (primary GPU)
        inputs = inputs.cuda(0)
        targets = targets.cuda(0)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.3f} | '
                  f'Acc: {100.*correct/total:.2f}% ({correct}/{total})')
    
    epoch_time = time.time() - start_time
    return running_loss / len(dataloader), 100. * correct / total, epoch_time

def evaluate_test(model, dataloader, criterion):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.cuda(0)  # Move to primary GPU
            targets = targets.cuda(0)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(dataloader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc

def plot_training_curves(results, save_path='multigpu_training_curves.png'):
    """Plot training and validation curves"""
    plt.figure(figsize=(15, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(results['train_history']['loss'], label='Train Loss')
    plt.plot(results['val_history']['loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(results['train_history']['acc'], label='Train Accuracy')
    plt.plot(results['val_history']['acc'], label='Val Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_batch_times(batch_times, save_path='multigpu_batch_times.png'):
    """Plot batch processing times"""
    plt.figure(figsize=(10, 5))
    plt.plot(batch_times)
    plt.title('Batch Processing Times')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    # Check if we have two GPUs available
    if torch.cuda.device_count() < 2:
        print("This script requires at least 2 GPUs to run!")
        return
    
    # Print GPU information
    print(f"Using GPUs:")
    for i in range(2):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load CIFAR-10
    print("\nLoading datasets...")
    full_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    # Split training set into train and validation
    train_size = 45000  # 90% of training data
    val_size = 5000    # 10% of training data
    trainset, valset = torch.utils.data.random_split(full_trainset, [train_size, val_size])
    
    # Load test set
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Create dataloaders
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    valloader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=4)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
    
    # Create model
    print("\nInitializing Multi-GPU AlexNet...")
    model = AlexNetMultiGPU(num_classes=10)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4
    )
    num_epochs = 20
    
    # Training history
    history = {
        'train_history': {'loss': [], 'acc': [], 'times': []},
        'val_history': {'loss': [], 'acc': []},
        'best_val_acc': 0.0,
        'best_epoch': 0
    }
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc, epoch_time = train_epoch(
            model, trainloader, criterion, optimizer, epoch
        )
        history['train_history']['loss'].append(train_loss)
        history['train_history']['acc'].append(train_acc)
        history['train_history']['times'].append(epoch_time)
        
        # Validate
        val_loss, val_acc = evaluate_test(model, valloader, criterion)
        history['val_history']['loss'].append(val_loss)
        history['val_history']['acc'].append(val_acc)
        
        # Track best model
        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            history['best_epoch'] = epoch
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'alexnet_multigpu_best.pth')
        
        print(f'\nEpoch {epoch}:')
        print(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.2f}%')
        print(f'Time: {epoch_time:.2f}s')
        
        # Save checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'alexnet_multigpu_checkpoint_epoch_{epoch}.pth')
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = evaluate_test(model, testloader, criterion)
    print(f'Test Loss: {test_loss:.3f}')
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Add test results to history
    history['test_results'] = {
        'loss': test_loss,
        'accuracy': test_acc
    }
    
    # Plot training curves
    print("\nGenerating plots...")
    plot_training_curves(history)
    plot_batch_times(history['train_history']['times'])
    
    # Save final results
    print("\nSaving final results...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, 'alexnet_multigpu_final.pth')
    
    # Print summary
    print("\nTraining Summary:")
    print(f"Best Validation Accuracy: {history['best_val_acc']:.2f}% (Epoch {history['best_epoch']})")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Average Epoch Time: {np.mean(history['train_history']['times']):.2f}s")
    print("\nSaved files:")
    print("- alexnet_multigpu_final.pth (Final model and full history)")
    print("- alexnet_multigpu_best.pth (Best model based on validation accuracy)")
    print("- multigpu_training_curves.png (Loss and accuracy curves)")
    print("- multigpu_batch_times.png (Training time per epoch)")

if __name__ == '__main__':
    main()
