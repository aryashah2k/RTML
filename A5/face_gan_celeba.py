import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import logging
import os
from tensorboardX import SummaryWriter
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('FaceGAN')

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Create directories for saving models and results
os.makedirs('models/face_gan', exist_ok=True)
os.makedirs('results/face_gan', exist_ok=True)
os.makedirs('data/celeba', exist_ok=True)

# Image parameters
image_size = 64
nc = 3  # Number of channels (RGB)
nz = 100  # Size of latent vector
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator

class Generator(nn.Module):
    """
    Generator network for DCGAN with improvements
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is latent vector z
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: ngf x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # State size: nc x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    """
    Discriminator network for DCGAN with improvements
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def weights_init(m):
    """
    Custom weights initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Logger:
    """
    Logger for the GAN training process
    """
    def __init__(self, model_name='face_gan'):
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
    
    def log_images(self, images, num_images, epoch, n_batch, num_batches):
        """Log images to TensorBoard"""
        step = epoch * num_batches + n_batch
        grid = vutils.make_grid(images, padding=2, normalize=True)
        self.writer.add_image(f'{self.model_name}/generated_images', grid, step)
        
        # Save images to disk
        img_dir = f'results/{self.model_name}/images'
        os.makedirs(img_dir, exist_ok=True)
        vutils.save_image(images, f'{img_dir}/epoch_{epoch}_batch_{n_batch}.png', 
                         normalize=True, padding=2)
    
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

def get_celeba_dataset(path):
    """
    Get the CelebA dataset from the specified path
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Use a custom dataset class to load images directly from the directory
    class CelebADataset(torch.utils.data.Dataset):
        def __init__(self, img_dir, transform=None):
            self.img_dir = img_dir
            self.transform = transform
            self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
            logger.info(f"Found {len(self.image_files)} images in {img_dir}")
            
        def __len__(self):
            return len(self.image_files)
            
        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.image_files[idx])
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, 0  # Return dummy label 0
    
    # Create dataset
    dataset = CelebADataset(path, transform=transform)
    logger.info(f"CelebA dataset loaded successfully with {len(dataset)} images")
    
    return dataset

def noise(batch_size):
    """Generate random noise for the generator"""
    return torch.randn(batch_size, nz, 1, 1, device=device)

def train_face_gan(num_epochs=50, batch_size=128, save_interval=5, log_interval=100, lr=0.0002, path='RTML/A5/data/celeba/img_align_celeba'):
    """Train the face GAN"""
    logger.info("Starting Face GAN training...")
    
    # Get dataset
    dataset = get_celeba_dataset(path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    num_batches = len(dataloader)
    
    # Create networks
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Setup optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Fixed noise for visualization
    fixed_noise = noise(64)
    
    # Create logger
    log = Logger()
    
    # Labels
    real_label = 0.9  # Label smoothing
    fake_label = 0
    
    # Start training
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with real batch
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            # Train with fake batch
            noise_input = noise(batch_size)
            fake = netG(noise_input)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            
            # Only update discriminator every other batch
            if i % 2 == 0:
                optimizerD.step()
            
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # Generator wants discriminator to think its output is real
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            # Log batch error
            log.log(errD, errG, output, output, epoch, i, num_batches)
            
            # Output training stats
            if i % log_interval == 0:
                log.display_status(
                    epoch, num_epochs, i, num_batches,
                    errD, errG, torch.tensor([D_x]), torch.tensor([D_G_z2])
                )
                
                # Generate and log images
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                log.log_images(fake, 64, epoch, i, num_batches)
        
        # Save models at specified intervals
        if epoch % save_interval == 0 or epoch == num_epochs - 1:
            log.save_models(netG, netD, epoch)
            
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")
    
    # Save final models
    log.save_models(netG, netD, num_epochs-1)
    
    # Plot training curves
    log.plot_losses()
    log.plot_scores()
    
    # Close logger
    log.close()
    
    total_time = time.time() - start_time
    logger.info(f"Face GAN training completed in {datetime.timedelta(seconds=total_time)}")
    
    return netG, netD

def generate_faces(generator, num_images=16, grid_size=4):
    """Generate and display faces using the trained generator"""
    logger.info(f"Generating {num_images} face images...")
    
    # Generate images
    with torch.no_grad():
        z = noise(num_images)
        generated_images = generator(z).detach().cpu()
    
    # Save individual images
    img_dir = f'results/face_gan/generated'
    os.makedirs(img_dir, exist_ok=True)
    
    for i in range(num_images):
        img = generated_images[i]
        img = img * 0.5 + 0.5  # Unnormalize
        vutils.save_image(img, f'{img_dir}/face_{i}.png')
    
    # Create and save grid
    grid = vutils.make_grid(generated_images, nrow=grid_size, padding=2, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Generated Faces")
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.savefig(f'results/face_gan/generated_faces_grid.png')
    plt.close()
    
    logger.info(f"Generated images saved to {img_dir}")
    logger.info(f"Grid image saved to results/face_gan/generated_faces_grid.png")

def interpolate_faces(generator, num_steps=10):
    """Generate a sequence of faces by interpolating between two random points in latent space"""
    logger.info("Generating face interpolation...")
    
    # Generate two random points in latent space
    with torch.no_grad():
        z1 = noise(1)
        z2 = noise(1)
        
        # Generate intermediate points
        images = []
        for alpha in np.linspace(0, 1, num_steps):
            z_interp = z1 * (1 - alpha) + z2 * alpha
            img = generator(z_interp).detach().cpu()
            images.append(img[0])
        
        # Create grid
        grid = vutils.make_grid(images, nrow=num_steps, padding=2, normalize=True)
        
        # Save grid
        plt.figure(figsize=(20, 5))
        plt.axis("off")
        plt.title("Face Interpolation")
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.savefig(f'results/face_gan/face_interpolation.png')
        plt.close()
        
        logger.info(f"Interpolation saved to results/face_gan/face_interpolation.png")

def load_model(model_path):
    """Load a trained generator model"""
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    return generator

if __name__ == "__main__":
    # Define dataset path - change this to your actual path
    dataset_path = './data/celeba/img_align_celeba'
    
    # Create directories for saving results
    os.makedirs('models/face_gan', exist_ok=True)
    os.makedirs('results/face_gan', exist_ok=True)
    os.makedirs('logs/face_gan', exist_ok=True)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA is not available. Using CPU for training (this will be slow).")
    
    # Train the face GAN
    try:
        generator, discriminator = train_face_gan(num_epochs=50, path=dataset_path)
        
        # Generate faces
        generate_faces(generator, num_images=16)
        
        # Generate face interpolation
        interpolate_faces(generator)
        
        logger.info("Training and generation completed successfully!")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())
