import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import logging
import os
from tensorboardX import SummaryWriter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('CustomGAN')

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Create directories for saving models and results
os.makedirs('models/custom_gan', exist_ok=True)
os.makedirs('results/custom_gan', exist_ok=True)

class CustomDataset(Dataset):
    """
    Dataset that generates 2D data according to the specified distribution:
    θ ~ U(0,2π)
    r ~ N(0,1)
    x = (10+r)cosθ
    y = (10+r)sinθ + 10 if π/2 ≤ θ ≤ 3π/2, otherwise (10+r)sinθ - 10
    """
    def __init__(self, size=10000):
        """Generate the dataset in the initialization"""
        self.size = size
        
        # Generate theta ~ U(0, 2π)
        theta = np.random.uniform(0, 2*np.pi, size)
        
        # Generate r ~ N(0, 1)
        r = np.random.normal(0, 1, size)
        
        # Calculate x coordinates: (10+r)cosθ
        x = (10 + r) * np.cos(theta)
        
        # Calculate y coordinates based on the condition
        y = np.zeros_like(r)
        mask = (theta >= np.pi/2) & (theta <= 3*np.pi/2)
        y[mask] = (10 + r[mask]) * np.sin(theta[mask]) + 10
        y[~mask] = (10 + r[~mask]) * np.sin(theta[~mask]) - 10
        
        # Store as tensor
        self.data = torch.tensor(np.column_stack((x, y)), dtype=torch.float32)
        
        logger.info(f"Generated dataset with {size} samples")
        
    def __len__(self):
        """Return the size of the dataset"""
        return self.size
    
    def __getitem__(self, idx):
        """Return a sample from the dataset"""
        return self.data[idx]
    
    def plot_samples(self, num_samples=1000, save_path=None):
        """Plot samples from the dataset"""
        indices = np.random.choice(len(self), min(num_samples, len(self)), replace=False)
        samples = self.data[indices].numpy()
        
        plt.figure(figsize=(10, 10))
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=10)
        plt.title("Data Distribution")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved plot to {save_path}")
        
        plt.close()
        return samples

class Generator(nn.Module):
    """
    Generator network for the custom GAN
    Input: Random noise vector of size 100
    Output: 2D data point
    """
    def __init__(self, input_size=100, hidden_size=128):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second hidden layer
            nn.Linear(hidden_size, hidden_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third hidden layer
            nn.Linear(hidden_size*2, hidden_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer
            nn.Linear(hidden_size*2, 2),
            # No activation at the end to allow for unbounded output values
        )
    
    def forward(self, z):
        """Forward pass"""
        return self.model(z)

class Discriminator(nn.Module):
    """
    Discriminator network for the custom GAN
    Input: 2D data point
    Output: Probability that the input is real
    """
    def __init__(self, input_size=2, hidden_size=128):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # Second hidden layer
            nn.Linear(hidden_size, hidden_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # Third hidden layer
            nn.Linear(hidden_size*2, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # Output layer
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)

class Logger:
    """
    Logger for the GAN training process
    """
    def __init__(self, model_name='custom_gan'):
        self.model_name = model_name
        self.writer = SummaryWriter(f'logs/{model_name}')
        self.d_losses = []
        self.g_losses = []
        self.d_real_scores = []
        self.d_fake_scores = []
        
    def log(self, d_loss, g_loss, d_real_score, d_fake_score, epoch, n_batch, num_batches):
        """Log losses and scores"""
        step = epoch * num_batches + n_batch
        
        # Log to TensorBoard
        self.writer.add_scalar(f'{self.model_name}/d_loss', d_loss.item(), step)
        self.writer.add_scalar(f'{self.model_name}/g_loss', g_loss.item(), step)
        self.writer.add_scalar(f'{self.model_name}/d_real_score', d_real_score.mean().item(), step)
        self.writer.add_scalar(f'{self.model_name}/d_fake_score', d_fake_score.mean().item(), step)
        
        # Store for plotting
        self.d_losses.append(d_loss.item())
        self.g_losses.append(g_loss.item())
        self.d_real_scores.append(d_real_score.mean().item())
        self.d_fake_scores.append(d_fake_score.mean().item())
    
    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_loss, g_loss, d_real_score, d_fake_score):
        """Display training status"""
        logger.info(f"Epoch: [{epoch}/{num_epochs}], Batch Num: [{n_batch}/{num_batches}]")
        logger.info(f"Discriminator Loss: {d_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}")
        logger.info(f"D(x): {d_real_score.mean().item():.4f}, D(G(z)): {d_fake_score.mean().item():.4f}")
    
    def save_models(self, generator, discriminator, epoch):
        """Save models"""
        torch.save(generator.state_dict(), f'models/{self.model_name}/generator_epoch_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'models/{self.model_name}/discriminator_epoch_{epoch}.pth')
        logger.info(f"Models saved at epoch {epoch}")
    
    def plot_losses(self):
        """Plot the training losses"""
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.g_losses, label="Generator")
        plt.plot(self.d_losses, label="Discriminator")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'results/{self.model_name}/loss_plot.png')
        plt.close()
        logger.info(f"Loss plot saved to results/{self.model_name}/loss_plot.png")
    
    def plot_scores(self):
        """Plot the discriminator scores"""
        plt.figure(figsize=(10, 5))
        plt.title("Discriminator Scores During Training")
        plt.plot(self.d_real_scores, label="D(x)")
        plt.plot(self.d_fake_scores, label="D(G(z))")
        plt.xlabel("Iterations")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig(f'results/{self.model_name}/score_plot.png')
        plt.close()
        logger.info(f"Score plot saved to results/{self.model_name}/score_plot.png")
    
    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()

def noise(batch_size, input_size=100):
    """Generate random noise for the generator"""
    return torch.randn(batch_size, input_size).to(device)

def real_data_target(size):
    """Target for real data"""
    return torch.ones(size, 1).to(device)

def fake_data_target(size):
    """Target for fake data"""
    return torch.zeros(size, 1).to(device)

def train_discriminator(discriminator, optimizer, real_data, fake_data, loss_fn):
    """Train the discriminator"""
    batch_size = real_data.size(0)
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Train on real data with label smoothing
    prediction_real = discriminator(real_data)
    real_target = real_data_target(batch_size) * 0.9  # Label smoothing
    error_real = loss_fn(prediction_real, real_target)
    error_real.backward()
    
    # Train on fake data
    prediction_fake = discriminator(fake_data)
    error_fake = loss_fn(prediction_fake, fake_data_target(batch_size))
    error_fake.backward()
    
    # Update weights
    optimizer.step()
    
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(discriminator, optimizer, fake_data, loss_fn):
    """Train the generator"""
    batch_size = fake_data.size(0)
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Generate fake data and calculate loss
    prediction = discriminator(fake_data)
    error = loss_fn(prediction, real_data_target(batch_size))
    error.backward()
    
    # Update weights
    optimizer.step()
    
    return error

def train_gan(dataset, num_epochs=200, batch_size=128, save_interval=20, log_interval=10):
    """Train the GAN"""
    logger.info("Starting GAN training on custom dataset...")
    
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_batches = len(data_loader)
    
    # Create networks
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Setup optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss = nn.BCELoss()
    
    # Create logger
    log = Logger()
    
    # Start training
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        for n_batch, real_batch in enumerate(data_loader):
            real_data = real_batch.to(device)
            batch_size = real_data.size(0)
            
            # Train discriminator
            # Only train discriminator every other batch to prevent it from becoming too strong
            if n_batch % 2 == 0:
                fake_data = generator(noise(batch_size)).detach()
                d_loss, d_pred_real, d_pred_fake = train_discriminator(
                    discriminator, d_optimizer, real_data, fake_data, loss
                )
            else:
                with torch.no_grad():
                    d_pred_real = discriminator(real_data)
                    fake_data = generator(noise(batch_size))
                    d_pred_fake = discriminator(fake_data)
                    d_loss = loss(d_pred_real, real_data_target(batch_size) * 0.9) + loss(d_pred_fake, fake_data_target(batch_size))
            
            # Train generator
            fake_data = generator(noise(batch_size))
            g_loss = train_generator(discriminator, g_optimizer, fake_data, loss)
            
            # Log batch error
            log.log(d_loss, g_loss, d_pred_real, d_pred_fake, epoch, n_batch, num_batches)
            
            # Display progress
            if n_batch % log_interval == 0:
                log.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_loss, g_loss, d_pred_real, d_pred_fake
                )
        
        # Save models at specified intervals
        if epoch % save_interval == 0 or epoch == num_epochs - 1:
            log.save_models(generator, discriminator, epoch)
            
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")
    
    # Save final models
    log.save_models(generator, discriminator, num_epochs-1)
    
    # Plot training curves
    log.plot_losses()
    log.plot_scores()
    
    # Close logger
    log.close()
    
    total_time = time.time() - start_time
    logger.info(f"GAN training completed in {datetime.timedelta(seconds=total_time)}")
    
    return generator, discriminator

def evaluate_gan(generator, dataset, num_samples=1000):
    """Evaluate the GAN by comparing real and generated samples"""
    # Generate samples
    with torch.no_grad():
        z = noise(num_samples)
        generated_samples = generator(z).cpu().numpy()
    
    # Get real samples
    real_samples = dataset.plot_samples(num_samples, save_path='results/custom_gan/real_samples.png')
    
    # Plot generated samples
    plt.figure(figsize=(10, 10))
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.6, s=10)
    plt.title("Generated Distribution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('results/custom_gan/generated_samples.png')
    plt.close()
    logger.info("Saved generated samples plot")
    
    # Plot comparison
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.6, s=10)
    plt.title("Real Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.6, s=10)
    plt.title("Generated Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('results/custom_gan/comparison.png')
    plt.close()
    logger.info("Saved comparison plot")

if __name__ == "__main__":
    # Create dataset
    dataset = CustomDataset(size=10000)
    
    # Plot some samples from the dataset
    dataset.plot_samples(save_path='results/custom_gan/dataset_samples.png')
    
    # Train GAN
    generator, discriminator = train_gan(dataset)
    
    # Evaluate GAN
    evaluate_gan(generator, dataset)
    
    logger.info("Done!")
