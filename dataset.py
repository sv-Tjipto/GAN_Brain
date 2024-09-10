# For loading and preprocessing the dataset.

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image

TRAIN_IMG = "/Volumes/sv_QuickFix/COMP3710/Assignment_2/GAN_Brain/Data_Brain/keras_png_slices_data/keras_png_slices_train"
VAL_IMG = "/Volumes/sv_QuickFix/COMP3710/Assignment_2/GAN_Brain/Data_Brain/keras_png_slices_data/keras_png_slices_validate"
TEST_IMG = "/Volumes/sv_QuickFix/COMP3710/Assignment_2/GAN_Brain/Data_Brain/keras_png_slices_data/keras_png_slices_test"


class OASISDataset(Dataset):
    """
    Custom dataset class for the OAS
    Lazy way to load the dataset, images are only loaded into memory when they are needed, one batch at a time.
    Memory efficient, but slower than loading the entire dataset into memory at once.
    """

    def __init__(self, image_dir_1,image_dir_2,image_dir_3, transform=None):
        # Combine the three image directories into one list
        # self.image_paths = sorted(os.listdir(image_dir_1)) + sorted(os.listdir(image_dir_2)) + sorted(os.listdir(image_dir_3))

         # Get image paths and filter out hidden files or non-image files
        self.image_paths = sorted([f for f in os.listdir(image_dir_1) if not f.startswith('._')]) + \
                           sorted([f for f in os.listdir(image_dir_2) if not f.startswith('._')]) + \
                           sorted([f for f in os.listdir(image_dir_3) if not f.startswith('._')])
        

        self.image_dir_1 = image_dir_1
        self.image_dir_2 = image_dir_2
        self.image_dir_3 = image_dir_3

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        # img_path = os.path.join(self.image_dir, self.image_paths[index])
        # image = Image.open(img_path).convert("L")  # Convert to grayscale
        
        # if self.transform:
        #     image = self.transform(image)
        
        # return image

        # Get the full path of the image
        img_name = self.image_paths[index]
        
        # Determine which folder the image belongs to
        if img_name in os.listdir(self.image_dir_1):
            img_path = os.path.join(self.image_dir_1, img_name)
        elif img_name in os.listdir(self.image_dir_2):
            img_path = os.path.join(self.image_dir_2, img_name)
        else:
            img_path = os.path.join(self.image_dir_3, img_name)
        
        # Load the image and convert to grayscale (or RGB depending on your data)
        image = Image.open(img_path).convert('L')  # 'L' for grayscale, use 'RGB' for color
        
        # Apply transformations (e.g., resizing, normalization)
        if self.transform:
            image = self.transform(image)
        
        return image

def load_data(batch_size):
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    # Define the dataset and dataloader
    image_dir = "/Volumes/sv_QuickFix/COMP3710/Assignment_2/GAN_Brain/Data_Brain"  # Path to OASIS dataset slices
    dataset = OASISDataset(TRAIN_IMG,VAL_IMG,TEST_IMG, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return dataloader