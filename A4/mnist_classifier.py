import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os
from A4_MAE import MAE_ViT, ViT_Classifier, setup_seed

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_classifier(pretrained_mae_path, patch_size, num_epochs=25, batch_size=128, device='cuda'):
    # Create directories for saving plots
    plot_dir = f'plots/mnist_classifier_patch{patch_size}'
    ensure_dir(plot_dir)
    
    # Data loading
    transform = Compose([
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Load pretrained MAE
    mae = MAE_ViT(
        image_size=28,
        patch_size=patch_size,
        emb_dim=192,
        encoder_layer=12,
        encoder_head=3,
        decoder_layer=4,
        decoder_head=3,
        mask_ratio=0.75
    ).to(device)
    
    checkpoint = torch.load(pretrained_mae_path, map_location=device)
    mae.load_state_dict(checkpoint['model_state_dict'])
    
    # Create classifier using pretrained encoder
    model = ViT_Classifier(mae.encoder, num_classes=10).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    best_acc = 0.0
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100.*correct/total
            })
        
        train_acc = 100.*correct/total
        train_losses.append(total_loss/len(train_loader))
        train_accs.append(train_acc)
        
        # Evaluation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = 100.*correct/total
        test_losses.append(test_loss/len(test_loader))
        test_accs.append(test_acc)
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_losses[-1]:.4f} | Test Acc: {test_acc:.2f}%')
        
        # Plot and save training curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epoch + 2), train_losses, 'b-', label='Train')
        plt.plot(range(1, epoch + 2), test_losses, 'r-', label='Test')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epoch + 2), train_accs, 'b-', label='Train')
        plt.plot(range(1, epoch + 2), test_accs, 'r-', label='Test')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'training_curves.png'))
        plt.close()
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            
            # Plot confusion matrix for best model
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix (Test Acc: {test_acc:.2f}%)')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
            plt.close()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, f'mnist_classifier_patch{patch_size}_best.pth')
    
    return model, best_acc

if __name__ == "__main__":
    setup_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train classifiers using different pretrained MAE models
    configs = [
        {'patch_size': 2, 'mask_ratio': 0.75},
        {'patch_size': 4, 'mask_ratio': 0.75},
        {'patch_size': 2, 'mask_ratio': 0.85},
        {'patch_size': 4, 'mask_ratio': 0.85},
    ]
    
    results = {}
    for config in configs:
        patch_size = config['patch_size']
        mask_ratio = config['mask_ratio']
        mae_path = f'mae_mnist_patch{patch_size}_mask{mask_ratio}_best.pth'
        
        print(f"\nTraining classifier using MAE with patch_size={patch_size}, mask_ratio={mask_ratio}")
        _, acc = train_classifier(mae_path, patch_size, device=device)
        results[f"patch{patch_size}_mask{mask_ratio}"] = acc
    
    # Print final results
    print("\nFinal Results:")
    for config, acc in results.items():
        print(f"{config}: {acc:.2f}%")
