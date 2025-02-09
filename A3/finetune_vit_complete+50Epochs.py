import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import random
from datetime import datetime
from tqdm import tqdm
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, x):
        return self.mha(x, x, x)[0]

class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
        )
        
    def forward(self, x):
        x1 = self.ln_1(x)
        attention_output = self.self_attention(x1, x1, x1)[0]
        x2 = x + attention_output
        x3 = self.ln_2(x2)
        x3 = x2 + self.mlp(x3)
        return x3

class ViT(nn.Module):
    def __init__(self, 
                 input_shape=(3, 224, 224),
                 patch_size=16,
                 hidden_dim=768,
                 num_heads=12,
                 num_layers=12,
                 mlp_dim=3072,
                 num_classes=100,
                 dropout=0.1):
        super().__init__()
        
        # Image and patch sizes
        channels, image_h, image_w = input_shape
        assert image_h % patch_size == 0 and image_w % patch_size == 0, 'Image dimensions must be divisible by patch size'
        self.patch_size = patch_size
        num_patches = (image_h // patch_size) * (image_w // patch_size)
        patch_dim = channels * patch_size * patch_size
        
        # Linear projection of flattened patches
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        
        # Dropout
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            EncoderBlock(hidden_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Classification head
        self.head = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize patch_embed like a linear layer
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)
        
        # Initialize cls_token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize the rest of the layers
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward(self, x):
        # Get dimensions
        B, C, H, W = x.shape
        
        # Reshape and flatten the image into patches
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p, p]
        x = x.flatten(1, 2)  # [B, H'*W', C, p, p]
        x = x.flatten(2)  # [B, H'*W', C*p*p]
        
        # Linear embedding
        x = self.patch_embed(x)  # [B, H'*W', hidden_dim]
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Take cls token output
        x = x[:, 0]
        
        # Classification
        x = self.head(x)
        
        return x

class SportDataset(Dataset):
    """Sport dataset with directory structure: train/val/test with class subfolders."""
    
    def __init__(self, csv_file, root_dir, split='train', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Root directory containing train/val/test folders.
            split (string): One of 'train', 'valid', or 'test'.
            transform (callable, optional): Optional transform to be applied on images.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Read the CSV file
        self.df = pd.read_csv(csv_file)
        
        # Filter by split
        self.df = self.df[self.df['data set'] == split]
        
        # Filter out non-image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.df = self.df[self.df['filepaths'].apply(
            lambda x: os.path.splitext(x)[1].lower() in valid_extensions
        )]
        
        # Create class to index mapping
        self.classes = sorted(self.df['labels'].unique())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        print(f"Loaded {split} split with {len(self.df)} images across {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get the image path and label
        img_path = os.path.join(self.root_dir, self.df.iloc[idx]['filepaths'])
        label = self.class_to_idx[self.df.iloc[idx]['labels']]
        
        # Load and convert image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a different valid image
            valid_indices = [i for i in range(len(self)) if i != idx]
            if not valid_indices:  # If no other images available
                raise RuntimeError("No valid images found in the dataset")
            return self.__getitem__(random.choice(valid_indices))
        
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error applying transform to {img_path}: {str(e)}")
                # Return a different valid image
                valid_indices = [i for i in range(len(self)) if i != idx]
                if not valid_indices:
                    raise RuntimeError("No valid images found in the dataset")
                return self.__getitem__(random.choice(valid_indices))
        
        return image, label

    def get_class_name(self, idx):
        """Get class name from class index."""
        return self.idx_to_class[idx]

class FineTuner:
    def __init__(self, model, device, checkpoint_path=None):
        self.device = device
        self.model = model.to(device)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f'runs/finetune_{timestamp}')
        
        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                print("Checkpoint structure:", type(checkpoint))
                if isinstance(checkpoint, dict):
                    print("Available keys in checkpoint:", checkpoint.keys())
                
                # Try different state dict keys that might be present
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint  # Assume it's the state dict itself
                else:
                    state_dict = checkpoint  # Assume it's the state dict itself
                
                print("\nModel's state dict keys:", self.model.state_dict().keys())
                print("\nCheckpoint's state dict keys:", state_dict.keys())
                
                # Try to load the state dict directly first
                try:
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                        print("Checkpoint loaded successfully with direct mapping!")
                        return
                    else:
                        print("\nDirect loading had issues:")
                        print("Missing keys:", missing_keys)
                        print("Unexpected keys:", unexpected_keys)
                except Exception as e:
                    print("\nDirect loading failed:", str(e))
                
                # If direct loading fails, try remapping the keys
                print("\nAttempting to remap keys...")
                new_state_dict = {}
                for k, v in state_dict.items():
                    # Handle different key names
                    if k == 'pos_embed':
                        new_state_dict['pos_embedding'] = v
                    elif k == 'cls_token':
                        new_state_dict['class_token'] = v
                    elif k == 'patch_embed.proj':
                        new_state_dict['conv_proj'] = v
                    elif 'blocks.' in k:
                        # Convert blocks.0.xxx to encoder.layers.encoder_layer_0.xxx
                        parts = k.split('.')
                        if len(parts) >= 2:
                            layer_num = parts[1]
                            rest_key = '.'.join(parts[2:])
                            new_key = f'encoder.layers.encoder_layer_{layer_num}.{rest_key}'
                            new_state_dict[new_key] = v
                    elif k == 'norm':
                        new_state_dict['encoder.ln'] = v
                    elif k == 'head':
                        new_state_dict['heads.0'] = v
                    else:
                        new_state_dict[k] = v
                
                # Try loading the remapped state dict
                try:
                    missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
                    print("\nAfter remapping:")
                    print("Missing keys:", missing_keys)
                    print("Unexpected keys:", unexpected_keys)
                    if len(missing_keys) == 0:
                        print("Checkpoint loaded successfully after remapping!")
                    else:
                        print("Warning: Some keys are still missing after remapping")
                except Exception as e:
                    print(f"Error loading remapped checkpoint: {str(e)}")
                    print("Starting with randomly initialized weights")
                    
            except Exception as e:
                print(f"Failed to load checkpoint: {str(e)}")
                print("Starting with randomly initialized weights")
    
    def train(self, train_loader, val_loader, epochs, lr=1e-4, weight_decay=1e-4):
        criterion = CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_acc = 0.0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Add gradient clipping
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': train_loss/(batch_idx+1), 
                    'acc': 100.*correct/total,
                    'lr': current_lr
                })
            
            train_loss = train_loss/len(train_loader)
            train_acc = 100.*correct/total
            
            # Validation
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f'\nEpoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, optimizer, val_acc, 'best_model.pth')
                print(f'Saved new best model with validation accuracy: {val_acc:.2f}%')
            
            # Early stopping check (optional)
            if val_acc > 95.0:  # You can adjust this threshold
                print(f'\nReached {val_acc:.2f}% validation accuracy. Stopping training.')
                break
        
        return train_losses, val_losses, train_accs, val_accs
    
    def evaluate(self, loader, criterion):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return val_loss/len(loader), 100.*correct/total
    
    def save_checkpoint(self, epoch, optimizer, val_acc, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }
        torch.save(checkpoint, filename)
    
    def predict(self, image_tensor):
        self.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            output = self.model(image_tensor)
            _, predicted = output.max(1)
            return predicted.item()
    
    def visualize_predictions(self, test_loader, class_names, num_images=8):
        images, labels = next(iter(test_loader))
        images = images[:num_images].to(self.device)
        labels = labels[:num_images]
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
            _, predictions = outputs.max(1)
        
        # Plot images with predictions
        fig = plt.figure(figsize=(15, 8))
        for idx in range(num_images):
            ax = fig.add_subplot(2, 4, idx+1, xticks=[], yticks=[])
            img = images[idx].cpu().numpy().transpose((1, 2, 0))
            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            
            pred_idx = predictions[idx].cpu().item()
            true_idx = labels[idx].item()
            
            pred_label = class_names[pred_idx] if pred_idx in class_names else f"Unknown ({pred_idx})"
            true_label = class_names[true_idx] if true_idx in class_names else f"Unknown ({true_idx})"
            
            color = 'green' if pred_idx == true_idx else 'red'
            ax.set_title(f'Pred: {pred_label}\nTrue: {true_label}', color=color)
        
        plt.tight_layout()
        return fig

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir='.'):
    """Create detailed training history plots."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss Over Time', fontsize=14, pad=10)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy Over Time', fontsize=14, pad=10)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_on_test_set(model, test_loader, device, class_names, save_dir='.'):
    """Evaluate model on test set and generate detailed report."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating on test set'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate accuracy
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Generate report
    report = f"Test Set Evaluation Report\n"
    report += f"{'='*50}\n"
    report += f"Overall Accuracy: {accuracy:.2f}%\n\n"
    
    # Per-class accuracy
    class_correct = {i: 0 for i in range(len(class_names))}
    class_total = {i: 0 for i in range(len(class_names))}
    
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    report += "Per-Class Performance:\n"
    report += f"{'-'*50}\n"
    report += f"{'Class':<30} {'Accuracy':<10} {'Samples':<10}\n"
    report += f"{'-'*50}\n"
    
    for class_idx in range(len(class_names)):
        if class_total[class_idx] > 0:
            class_acc = 100 * class_correct[class_idx] / class_total[class_idx]
            report += f"{class_names[class_idx]:<30} {class_acc:>8.2f}% {class_total[class_idx]:>10}\n"
    
    # Save report
    with open(os.path.join(save_dir, 'test_report.txt'), 'w') as f:
        f.write(report)
    
    return accuracy, report

if __name__ == '__main__':
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create output directory for results
    output_dir = f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = SportDataset(
        csv_file='sports-dataset/sports.csv',
        root_dir='sports-dataset',
        split='train',
        transform=train_transform
    )
    
    val_dataset = SportDataset(
        csv_file='sports-dataset/sports.csv',
        root_dir='sports-dataset',
        split='valid',
        transform=eval_transform
    )
    
    test_dataset = SportDataset(
        csv_file='sports-dataset/sports.csv',
        root_dir='sports-dataset',
        split='test',
        transform=eval_transform
    )
    
    if len(val_dataset) == 0:
        print("Warning: Validation dataset is empty! Please check the data split names in your CSV.")
        exit(1)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Initialize model
    model = ViT(
        input_shape=(3, 224, 224),
        patch_size=16,
        hidden_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_dim=3072,
        num_classes=len(train_dataset.classes)
    )
    
    # Print model summary
    print("\nModel Configuration:")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Input shape: (3, 224, 224)")
    print(f"Patch size: 16")
    print(f"Hidden dimension: 768")
    print(f"Number of heads: 12")
    print(f"Number of layers: 12")
    print(f"MLP dimension: 3072")
    
    # Initialize trainer with checkpoint
    trainer = FineTuner(model, device, checkpoint_path='best_model_epoch_50.pth')
    
    # Training parameters
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-4
    
    print("\nTraining Configuration:")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Batch size: 32")
    
    # Train the model
    train_losses, val_losses, train_accs, val_accs = trainer.train(
        train_loader, 
        val_loader, 
        num_epochs, 
        learning_rate, 
        weight_decay
    )
    
    # Plot and save training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs, output_dir)
    
    # Load best model for evaluation
    best_model_path = 'best_model.pth'
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model from epoch {checkpoint['epoch']} "
              f"with validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Evaluate on test set
    test_accuracy, test_report = evaluate_on_test_set(
        model, test_loader, device, 
        train_dataset.idx_to_class, output_dir
    )
    print("\nTest Set Evaluation:")
    print(test_report)
    
    # Visualize some predictions
    pred_fig = trainer.visualize_predictions(test_loader, train_dataset.idx_to_class, num_images=8)
    pred_fig.savefig(os.path.join(output_dir, 'test_predictions.png'))
    
    print(f"\nTraining completed! Results saved in '{output_dir}' directory.")
    print("Check the following files:")
    print(f"1. {output_dir}/training_history.png - Training progress plots")
    print(f"2. {output_dir}/test_report.txt - Detailed test set evaluation")
    print(f"3. {output_dir}/test_predictions.png - Sample predictions visualization")
