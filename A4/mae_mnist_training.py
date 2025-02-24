import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from A4_MAE import MAE_ViT, setup_seed, denormalize
import numpy as np

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_mae(patch_size, mask_ratio, num_epochs=25, batch_size=128, device='cuda'):
    # Create directories for saving plots
    plot_dir = f'plots/mae_mnist_patch{patch_size}_mask{mask_ratio}'
    ensure_dir(plot_dir)
    ensure_dir(os.path.join(plot_dir, 'reconstructions'))
    
    # Data loading
    transform = Compose([
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Model setup
    model = MAE_ViT(
        image_size=28,  # MNIST is 28x28
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
    losses = []  # Store losses for plotting
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            
            # Forward pass
            reconstructed, _ = model(images)
            loss = criterion(reconstructed, images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Save reconstruction samples every 200 batches
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
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f'mae_mnist_patch{patch_size}_mask{mask_ratio}_best.pth')
    
    return model

if __name__ == "__main__":
    setup_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Experiment with different configurations
    configs = [
        {'patch_size': 2, 'mask_ratio': 0.75},  # baseline
        {'patch_size': 4, 'mask_ratio': 0.75},  # larger patches
        {'patch_size': 2, 'mask_ratio': 0.85},  # more masking
        {'patch_size': 4, 'mask_ratio': 0.85},  # larger patches + more masking
    ]
    
    for config in configs:
        print(f"\nTraining MAE with patch_size={config['patch_size']}, mask_ratio={config['mask_ratio']}")
        model = train_mae(
            patch_size=config['patch_size'],
            mask_ratio=config['mask_ratio'],
            device=device
        )
