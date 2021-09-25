import os
import numpy as np
from PIL import Image
from torch.utils import data

class StyleTransferDataset(data.Dataset):
    def __init__(
        self,
        content_images_path,
        style_images_path,
        transform,
        n=2000
    ):            
        # Dataset Content
        self.content_images_path = content_images_path

        # Dataset Style
        self.style_images_path = style_images_path

        self.transform = transform

        self.n = n

    def __len__(self):
        """__len__"""
        return self.n

    def __getitem__(self, index):
        try:
            return self.getitem(index)
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self)))

    def getitem(self, index):
        content_path = np.random.choice(self.content_images_path)
        style_path = np.random.choice(self.style_images_path)

        content_im = Image.open(content_path).convert('RGB')
        style_im = Image.open(style_path).convert('RGB')

        return (
            self.transform(content_im),
            self.transform(style_im)
        )