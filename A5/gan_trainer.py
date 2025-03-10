#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import errno
from IPython import display
import logging
import time
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gan_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GAN_Trainer")

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Create directories for saving results
def make_directory(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# Create necessary directories
make_directory('./results/vanilla_gan/mnist')
make_directory('./results/dcgan/cifar')
make_directory('./models/vanilla_gan/mnist')
make_directory('./models/dcgan/cifar')
make_directory('./plots/vanilla_gan/mnist')
make_directory('./plots/dcgan/cifar')

class Logger:
    """
    Logger class for tracking training progress and visualizing results
    """
    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name
        self.comment = f'{model_name}_{data_name}'
        self.data_subdir = f'{model_name}/{data_name}'
        
        # TensorBoard
        self.writer = SummaryWriter(comment=self.comment)
        
        # Lists to store loss values
        self.d_losses = []
        self.g_losses = []
        self.epochs = []
        self.d_x_values = []  # D(x) - discriminator output for real data
        self.d_g_z_values = []  # D(G(z)) - discriminator output for fake data

    def log(self, d_error, g_error, epoch, n_batch, num_batches, d_pred_real=None, d_pred_fake=None):
        """Log losses and discriminator outputs to TensorBoard"""
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()
            
        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(f'{self.comment}/D_error', d_error, step)
        self.writer.add_scalar(f'{self.comment}/G_error', g_error, step)
        
        # Store values for later plotting
        if n_batch == num_batches - 1:  # Only store once per epoch
            self.d_losses.append(d_error)
            self.g_losses.append(g_error)
            self.epochs.append(epoch)
            
            if d_pred_real is not None and d_pred_fake is not None:
                d_x = d_pred_real.mean().item()
                d_g_z = d_pred_fake.mean().item()
                self.d_x_values.append(d_x)
                self.d_g_z_values.append(d_g_z)
                self.writer.add_scalar(f'{self.comment}/D(x)', d_x, step)
                self.writer.add_scalar(f'{self.comment}/D(G(z))', d_g_z, step)

    def log_images(self, images, num_images, epoch, n_batch, num_batches, format='NCHW', normalize=True):
        """Log images to TensorBoard and save them to disk"""
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)
        
        if format == 'NHWC':
            images = images.transpose(1, 3)
        
        step = Logger._step(epoch, n_batch, num_batches)
        img_name = f'{self.comment}/images'
        
        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(images, normalize=normalize, scale_each=True)
        
        # Make vertical grid from image tensor
        nrows = int(np.sqrt(num_images))
        grid = vutils.make_grid(images, nrow=nrows, normalize=True, scale_each=True)
        
        # Add horizontal images to tensorboard
        self.writer.add_image(img_name, horizontal_grid, step)
        
        # Save plots
        self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True):
        """Save images to disk"""
        out_dir = f'./results/{self.data_subdir}'
        Logger._make_dir(out_dir)
        
        # Plot and save horizontal
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        self._save_images(fig, epoch, n_batch, 'hori')
        plt.close()
        
        # Save squared
        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch)
        plt.close()

    def _save_images(self, fig, epoch, n_batch, comment=''):
        """Save image figures to disk"""
        out_dir = f'./results/{self.data_subdir}'
        Logger._make_dir(out_dir)
        fig.savefig(f'{out_dir}/{comment}_epoch_{epoch}_batch_{n_batch}.png')

    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):
        """Display training status"""
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()
        if isinstance(d_pred_real, torch.autograd.Variable):
            d_pred_real = d_pred_real.data
        if isinstance(d_pred_fake, torch.autograd.Variable):
            d_pred_fake = d_pred_fake.data
        
        logger.info(f'Epoch: [{epoch}/{num_epochs}], Batch Num: [{n_batch}/{num_batches}]')
        logger.info(f'Discriminator Loss: {d_error:.4f}, Generator Loss: {g_error:.4f}')
        logger.info(f'D(x): {d_pred_real.mean():.4f}, D(G(z)): {d_pred_fake.mean():.4f}')

    def save_models(self, generator, discriminator, epoch):
        """Save models to disk"""
        out_dir = f'./models/{self.data_subdir}'
        Logger._make_dir(out_dir)
        torch.save(generator.state_dict(), f'{out_dir}/G_epoch_{epoch}')
        torch.save(discriminator.state_dict(), f'{out_dir}/D_epoch_{epoch}')
        
    def plot_losses(self):
        """Plot discriminator and generator losses"""
        plt.figure(figsize=(10, 5))
        plt.title(f"Generator and Discriminator Loss During Training of {self.model_name} on {self.data_name}")
        plt.plot(self.epochs, self.g_losses, label="Generator")
        plt.plot(self.epochs, self.d_losses, label="Discriminator")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'./plots/{self.data_subdir}/loss_plot.png')
        logger.info(f"Loss plot saved to ./plots/{self.data_subdir}/loss_plot.png")
        
    def plot_discriminator_scores(self):
        """Plot discriminator scores for real and fake images"""
        plt.figure(figsize=(10, 5))
        plt.title(f"Discriminator scores for real and fake images - {self.model_name} on {self.data_name}")
        plt.plot(self.epochs, self.d_x_values, label="D(x) - Real")
        plt.plot(self.epochs, self.d_g_z_values, label="D(G(z)) - Fake")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig(f'./plots/{self.data_subdir}/discriminator_scores.png')
        logger.info(f"Discriminator scores plot saved to ./plots/{self.data_subdir}/discriminator_scores.png")

    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        """Calculate step for TensorBoard logging"""
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        """Create directory if it doesn't exist"""
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

# Data loading functions
def mnist_data():
    """Load MNIST dataset"""
    compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    out_dir = './data/mnist'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

def cifar_data():
    """Load CIFAR-10 dataset"""
    compose = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    out_dir = './data/cifar'
    return datasets.CIFAR10(root=out_dir, train=True, transform=compose, download=True)

# Utility functions
def images_to_vectors(images):
    """Convert MNIST images to vectors"""
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    """Convert vectors to MNIST images"""
    return vectors.view(vectors.size(0), 1, 28, 28)

def noise(size, features=100):
    """Generate noise vectors"""
    n = torch.randn(size, features, device=device)
    return n

def real_data_target(size):
    """Generate target tensor with ones"""
    return torch.ones(size, 1, device=device)

def fake_data_target(size):
    """Generate target tensor with zeros"""
    return torch.zeros(size, 1, device=device)

# Vanilla GAN for MNIST
class GeneratorNet(nn.Module):
    """
    A three hidden-layer generative neural network for MNIST
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

class DiscriminatorNet(nn.Module):
    """
    A three hidden-layer discriminative neural network for MNIST
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 1
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.Linear(256, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

# DCGAN for CIFAR-10
class DiscriminativeNet(nn.Module):
    """
    Discriminator network for DCGAN on CIFAR-10
    """
    def __init__(self):
        super(DiscriminativeNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(1024*4*4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 1024*4*4)
        x = self.out(x)
        return x

class GenerativeNet(nn.Module):
    """
    Generator network for DCGAN on CIFAR-10
    """
    def __init__(self):
        super(GenerativeNet, self).__init__()
        
        self.linear = nn.Linear(100, 1024*4*4)
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=3, kernel_size=4,
                stride=2, padding=1, bias=False
            )
        )
        self.out = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 1024, 4, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.out(x)
        return x

def init_weights(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# Training functions
def train_discriminator(discriminator, optimizer, real_data, fake_data, loss_fn):
    """Train the discriminator on a batch of real and fake data"""
    batch_size = real_data.size(0)
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Train on real data with label smoothing (0.9 instead of 1.0)
    prediction_real = discriminator(real_data)
    # Use label smoothing: target is 0.9 instead of 1.0
    real_target = real_data_target(batch_size) * 0.9
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
    """Train the generator to fool the discriminator"""
    batch_size = fake_data.size(0)
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    
    # Calculate error and backpropagate
    error = loss_fn(prediction, real_data_target(batch_size))
    error.backward()
    
    # Update weights
    optimizer.step()
    
    return error

def train_vanilla_gan_mnist(num_epochs=50, batch_size=100, save_interval=5, log_interval=100):
    """Train a vanilla GAN on MNIST dataset"""
    logger.info("Starting Vanilla GAN training on MNIST...")
    
    # Load data
    data = mnist_data()
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    num_batches = len(data_loader)
    
    # Create networks
    discriminator = DiscriminatorNet().to(device)
    generator = GeneratorNet().to(device)
    
    # Setup optimization
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss = nn.BCELoss()
    
    # Create logger
    log = Logger(model_name='vanilla_gan', data_name='mnist')
    
    # Create fixed noise for visualization
    num_test_samples = 16
    test_noise = noise(num_test_samples)
    
    # Start training
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        for n_batch, (real_batch, _) in enumerate(data_loader):
            # Train discriminator
            real_data = images_to_vectors(real_batch).to(device)
            fake_data = generator(noise(real_data.size(0))).detach()
            d_error, d_pred_real, d_pred_fake = train_discriminator(
                discriminator, d_optimizer, real_data, fake_data, loss
            )
            
            # Train generator
            fake_data = generator(noise(real_batch.size(0)))
            g_error = train_generator(discriminator, g_optimizer, fake_data, loss)
            
            # Log batch error
            log.log(d_error, g_error, epoch, n_batch, num_batches, d_pred_real, d_pred_fake)
            
            # Display progress
            if (n_batch) % log_interval == 0:
                # Generate and log images
                test_images = vectors_to_images(generator(test_noise))
                test_images = test_images.data.cpu()
                log.log_images(test_images, num_test_samples, epoch, n_batch, num_batches)
                
                # Display status
                log.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
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
    log.plot_discriminator_scores()
    
    # Close logger
    log.close()
    
    total_time = time.time() - start_time
    logger.info(f"Vanilla GAN training completed in {datetime.timedelta(seconds=total_time)}")
    
    return generator, discriminator

def train_dcgan_cifar(num_epochs=30, batch_size=100, save_interval=5, log_interval=100):
    """Train a DCGAN on CIFAR-10 dataset"""
    logger.info("Starting DCGAN training on CIFAR-10...")
    
    # Load data
    data = cifar_data()
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    num_batches = len(data_loader)
    
    # Create networks
    discriminator = DiscriminativeNet().to(device)
    generator = GenerativeNet().to(device)
    
    # Initialize weights
    discriminator.apply(init_weights)
    generator.apply(init_weights)
    
    # Setup optimization with different learning rates
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))  # Lower learning rate
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss = nn.BCELoss()
    
    # Create logger
    log = Logger(model_name='dcgan', data_name='cifar')
    
    # Create fixed noise for visualization
    num_test_samples = 16
    test_noise = noise(num_test_samples)
    
    # Start training
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        for n_batch, (real_batch, _) in enumerate(data_loader):
            real_data = real_batch.to(device)
            batch_size = real_batch.size(0)
            
            # Train discriminator only every 2nd iteration to prevent it from becoming too strong
            if n_batch % 2 == 0:
                # Generate fake data
                fake_data = generator(noise(batch_size)).detach()
                
                # Train discriminator
                d_error, d_pred_real, d_pred_fake = train_discriminator(
                    discriminator, d_optimizer, real_data, fake_data, loss
                )
            else:
                # Set these for logging when we skip discriminator training
                with torch.no_grad():
                    d_pred_real = discriminator(real_data)
                    fake_data = generator(noise(batch_size))
                    d_pred_fake = discriminator(fake_data)
                    d_error = loss(d_pred_real, real_data_target(batch_size) * 0.9) + loss(d_pred_fake, fake_data_target(batch_size))
            
            # Train generator
            fake_data = generator(noise(batch_size))
            g_error = train_generator(discriminator, g_optimizer, fake_data, loss)
            
            # Log batch error
            log.log(d_error, g_error, epoch, n_batch, num_batches, d_pred_real, d_pred_fake)
            
            # Display progress
            if (n_batch) % log_interval == 0:
                # Generate and log images
                test_images = generator(test_noise)
                test_images = test_images.data.cpu()
                log.log_images(test_images, num_test_samples, epoch, n_batch, num_batches)
                
                # Display status
                log.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
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
    log.plot_discriminator_scores()
    
    # Close logger
    log.close()
    
    total_time = time.time() - start_time
    logger.info(f"DCGAN training completed in {datetime.timedelta(seconds=total_time)}")
    
    return generator, discriminator

def test_generator(generator, model_type='vanilla_gan', dataset='mnist', num_images=16):
    """Test a trained generator by generating images"""
    logger.info(f"Testing {model_type} generator on {dataset}...")
    
    # Create noise for generation
    z_dim = 100
    test_noise = noise(num_images)
    
    # Set generator to evaluation mode
    generator.eval()
    
    with torch.no_grad():
        # Generate images
        generated_images = generator(test_noise)
        
        # Convert vectors to images for vanilla GAN MNIST
        if model_type == 'vanilla_gan' and dataset == 'mnist':
            generated_images = vectors_to_images(generated_images)
        
        # Save images
        out_dir = f'./results/{model_type}/{dataset}/test'
        os.makedirs(out_dir, exist_ok=True)
        
        # Make grid of images
        img_grid = vutils.make_grid(generated_images.cpu(), normalize=True, nrow=4)
        
        # Save grid
        plt.figure(figsize=(10, 10))
        plt.imshow(np.transpose(img_grid.numpy(), (1, 2, 0)))
        plt.axis('off')
        plt.title(f'Generated Images - {model_type} on {dataset}')
        plt.savefig(f'{out_dir}/generated_images.png')
        
        logger.info(f"Generated images saved to {out_dir}/generated_images.png")
        
        # Save individual images
        for i, img in enumerate(generated_images):
            vutils.save_image(img.cpu(), f'{out_dir}/image_{i}.png', normalize=True)

if __name__ == '__main__':
    # Set number of epochs for each model
    vanilla_gan_epochs = 30
    dcgan_epochs = 30
    
    # Train Vanilla GAN on MNIST
   # vanilla_generator, vanilla_discriminator = train_vanilla_gan_mnist(
    #    num_epochs=vanilla_gan_epochs,
     #   batch_size=100,
      #  save_interval=5,
       # log_interval=100
    #)
    
    # Test Vanilla GAN
    #test_generator(vanilla_generator, model_type='vanilla_gan', dataset='mnist')
    
    # Train DCGAN on CIFAR-10
    dcgan_generator, dcgan_discriminator = train_dcgan_cifar(
        num_epochs=dcgan_epochs,
        batch_size=100,
        save_interval=5,
        log_interval=100
    )
    
    # Test DCGAN
    test_generator(dcgan_generator, model_type='dcgan', dataset='cifar')
    
    logger.info("All training and testing completed successfully!")
