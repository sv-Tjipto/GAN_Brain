# For the training loop and managing the training process.

import torch
import torch.nn as nn
from dataset import load_data
from gan import Generator, Discriminator
# from gan import get_optimizers, create_noise
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.utils import make_grid
from tqdm import tqdm # For displaying progress bar
import os

TRAIN = False
EPOCH_VISUALIZATION = False
TEST = True
LOAD_CHECKPOINT = False
CHECKPOINT_PATH = 'checkpoints/gan_brain_checkpoint_epoch.pth'
CHECKPOINT_DIR = 'checkpoints'

def devicer():
    # Prioritize CUDA, then MPS, and finally CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f'Using device: {device}', flush=True)
    return device

def generate_images(generator, num_images=20, noise_dim=100, device='mps'):
    generator.eval()  # Set the generator to evaluation mode
    
    sample_noise = torch.randn(num_images, noise_dim).to(device)
    generated_images = generator(sample_noise).detach().cpu()
    generated_images = (generated_images + 1) / 2  # Rescale to [0, 1] for visualization
    grid = make_grid(generated_images, nrow=4)
    plt.imshow(grid.permute(1, 2, 0), cmap="gray")
    plt.show()



def plot_images(images):
    num_images = len(images)
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()

def load_checkpoint(checkpoint_path, generator, discriminator, gen_optimizer, disc_optimizer):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Load the saved states into the models and optimizers
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
    disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])

    # Load the last saved epoch
    start_epoch = checkpoint['epoch']

    print(f"Checkpoint loaded from epoch {start_epoch}")
    return start_epoch


# def train1(generator, discriminator, adversarial_loss,optimizer_G, optimizer_D, dataloader, latent_dim, epochs, epoch, batch_size):
    
    # Initialize tqdm progress bar
    progress_bar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")

    for i, imgs in enumerate(dataloader):
        real_imgs = imgs.to(device)
        batch_size = real_imgs.size(0)
        
        # Labels for real (1) and fake (0) images
        real = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # -----------------
        # Train Discriminator
        # -----------------

        # Sample noise as generator input
        z = torch.randn(batch_size, latent_dim).to(device)

        # Generate fake images
        gen_imgs = generator(z)

        # Loss for real and fake images
        real_loss = adversarial_loss(discriminator(real_imgs), real)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        # Backprop and optimize discriminator
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        # Train Generator
        # -----------------

        # Train the generator to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), real)  # We want fake images to be classified as real

        # Backprop and optimize generator
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # Print progress
        # print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # Update tqdm progress bar with losses
        progress_bar.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

    # Optionally, generate and save some example images after each epoch
    if EPOCH_VISUALIZATION:
        with torch.no_grad():
            sample_noise = torch.randn(16, latent_dim).to(device)
            generated_images = generator(sample_noise).detach().cpu()
            generated_images = (generated_images + 1) / 2  # Rescale to [0, 1] for visualization
            grid = make_grid(generated_images, nrow=4)
            plt.imshow(grid.permute(1, 2, 0), cmap="gray")
            plt.show()

def save_checkpoint(generator, discriminator, gen_optimizer, disc_optimizer, epoch, checkpoint_dir=CHECKPOINT_DIR):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'gen_optimizer_state_dict': gen_optimizer.state_dict(),
        'disc_optimizer_state_dict': disc_optimizer.state_dict(),
    }
    
    torch.save(checkpoint, f"{checkpoint_dir}/gan_brain_checkpoint_epoch.pth")
    print(f"Checkpoint saved at epoch {epoch}.", flush =True)

def train(generator, discriminator, adversarial_loss, optimizer_G, optimizer_D, dataloader, latent_dim, epochs, epoch, batch_size, device, visualize=False):
    
    # Initialize tqdm progress bar
    progress_bar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}", leave=True, dynamic_ncols=True)

    for i, real_imgs in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        
        # Labels for real (1) and fake (0) images
        real = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # -----------------
        # Train Discriminator
        # -----------------

        # Sample noise as generator input
        z = torch.randn(batch_size, latent_dim).to(device)

        # Generate fake images
        gen_imgs = generator(z)

        # Loss for real and fake images
        real_loss = adversarial_loss(discriminator(real_imgs), real)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        # Backprop and optimize discriminator
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        # Train Generator
        # -----------------

        # Train the generator to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), real)  # We want fake images to be classified as real

        # Backprop and optimize generator
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # Update tqdm progress bar with losses
        progress_bar.set_postfix(d_loss=f"{d_loss.item():.4f}", g_loss=f"{g_loss.item():.4f}")
        
        # Ensure the progress bar is updated regularly
        progress_bar.update(1)

    # Optionally, generate and save some example images after each epoch
    if visualize and epoch+1 % 5 == 0:
        with torch.no_grad():
            sample_noise = torch.randn(16, latent_dim).to(device)
            generated_images = generator(sample_noise).detach().cpu()
            generated_images = (generated_images + 1) / 2  # Rescale to [0, 1] for visualization
            grid = make_grid(generated_images, nrow=4)
            plt.imshow(grid.permute(1, 2, 0), cmap="gray")
            plt.show()

if __name__ == '__main__':
    # Hyperparameters
    batch_size = 128  # Number of images in each batch
    latent_dim = 100  # Dimension of the latent noise vector
    epochs = 100  # Number of training epochs

    device = devicer()

    # Initialize generator and discriminator
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # Load the dataset
    dataloader = load_data(batch_size)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss function
    adversarial_loss = nn.BCELoss()

    if TRAIN:

        if LOAD_CHECKPOINT:
            start_epoch = load_checkpoint(CHECKPOINT_PATH, generator, discriminator, optimizer_G, optimizer_D)
        # Train the GAN
        for epoch in range(epochs):
            
            train(generator, discriminator, adversarial_loss,optimizer_G, optimizer_D, dataloader, latent_dim, epochs,epoch, batch_size, device, visualize=EPOCH_VISUALIZATION)
            if epoch % 10 == 0:
                save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch)
    
    if TEST:
        # generator = build_generator().to(device)  # Recreate the generator structure
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('mps'))
        generator.load_state_dict(checkpoint['generator_state_dict'])  # Load the saved model

        # Use the generator to create new images
        generate_images(generator, num_images=16, noise_dim=100, device=device)

        # # Plot the images
        # plot_images(fake_images)