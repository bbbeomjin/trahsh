import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MNIST(Dataset):


    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

       
        self.image_files = [file for file in os.listdir(data_dir) if file.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label = int(img_name.split('_')[1].split('.')[0])

        return image, label