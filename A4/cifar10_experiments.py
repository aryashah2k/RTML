import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os
from A4_MAE import MAE_ViT, ViT_Classifier, setup_seed, denormalize

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_mae_cifar10(patch_size, mask_ratio, num_epochs=50, batch_size=128, device='cuda'):
    # Create directories for saving plots
    plot_dir = f'plots/mae_cifar10_patch{patch_size}_mask{mask_ratio}'
    ensure_dir(plot_dir)
    ensure_dir(os.path.join(plot_dir, 'reconstructions'))
    
    # Data loading with augmentation
    transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    train_dataset = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Model setup - modified for RGB images
    model = MAE_ViT(
        image_size=32,
        patch_size=patch_size,
        emb_dim=192,
        encoder_layer=12,
        encoder_head=3,
        decoder_layer=4,
        decoder_head=3,
        mask_ratio=mask_ratio
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
    criterion = nn.MSELoss()
    
    # Training loop
    best_loss = float('inf')
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            
            # Convert RGB to grayscale for current implementation
            images = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
            
            reconstructed, _ = model(images)
            loss = criterion(reconstructed, images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            if batch_idx % 200 == 0:
                with torch.no_grad():
                    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
                    for i in range(4):
                        reconstructed_vis, _ = model(images[i:i+1])
                        orig_img = denormalize(images[i].cpu())
                        recon_img = denormalize(reconstructed_vis[0].cpu())
                        
                        axes[0, i].imshow(orig_img.squeeze(), cmap='gray')
                        axes[0, i].axis('off')
                        axes[0, i].set_title('Original')
                        
                        axes[1, i].imshow(recon_img.squeeze(), cmap='gray')
                        axes[1, i].axis('off')
                        axes[1, i].set_title('Reconstructed')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, 'reconstructions', f'epoch_{epoch}_batch_{batch_idx}.png'))
                    plt.close()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # Plot and save loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch + 2), losses, 'b-')
        plt.title(f'Training Loss (patch_size={patch_size}, mask_ratio={mask_ratio})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'loss_curve.png'))
        plt.close()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f'mae_cifar10_patch{patch_size}_mask{mask_ratio}_best.pth')
    
    return model

def train_classifier_cifar10(pretrained_mae_path, patch_size, num_epochs=30, batch_size=128, device='cuda'):
    # Create directories for saving plots
    plot_dir = f'plots/cifar10_classifier_patch{patch_size}'
    ensure_dir(plot_dir)
    
    # Data loading with augmentation
    transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    transform_test = Compose([
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    train_dataset = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    test_dataset = CIFAR10(root='./data', train=False, transform=transform_test, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Load pretrained MAE
    mae = MAE_ViT(
        image_size=32,
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
    
    model = ViT_Classifier(mae.encoder, num_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
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
            
            # Convert RGB to grayscale
            images = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
            
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
        
        scheduler.step()
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
                images = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
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
        
        # Save best model and confusion matrix
        if test_acc > best_acc:
            best_acc = test_acc
            
            # Plot confusion matrix for best model
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                         'dog', 'frog', 'horse', 'ship', 'truck']
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix (Test Acc: {test_acc:.2f}%)')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
            plt.close()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, f'cifar10_classifier_patch{patch_size}_best.pth')
    
    return model, best_acc

if __name__ == "__main__":
    setup_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Experiment configurations
    configs = [
        {'patch_size': 4, 'mask_ratio': 0.75},  # baseline for CIFAR-10
        {'patch_size': 8, 'mask_ratio': 0.75},  # larger patches
        {'patch_size': 4, 'mask_ratio': 0.85},  # more masking
    ]
    
    # Train MAE models
    for config in configs:
        print(f"\nTraining MAE with patch_size={config['patch_size']}, mask_ratio={config['mask_ratio']}")
        train_mae_cifar10(
            patch_size=config['patch_size'],
            mask_ratio=config['mask_ratio'],
            device=device
        )
    
    # Train classifiers
    results = {}
    for config in configs:
        patch_size = config['patch_size']
        mask_ratio = config['mask_ratio']
        mae_path = f'mae_cifar10_patch{patch_size}_mask{mask_ratio}_best.pth'
        
        print(f"\nTraining classifier using MAE with patch_size={patch_size}, mask_ratio={mask_ratio}")
        _, acc = train_classifier_cifar10(mae_path, patch_size, device=device)
        results[f"patch{patch_size}_mask{mask_ratio}"] = acc
    
    # Print final results
    print("\nFinal Results:")
    for config, acc in results.items():
        print(f"{config}: {acc:.2f}%")
