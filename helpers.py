import torch
from torch.utils.data import Dataset
from PIL import Image
import os


# Dataset loader

class ImageDataset(Dataset):
    def __init__(self, root_dir, annotations, transform=None, labels=True):
        self.root_dir = root_dir
        self.annotations = annotations
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_id = self.annotations.iloc[index, 1]
        # print(image_id)
        # print(type(image_id))
        img = Image.open(os.path.join(self.root_dir, image_id)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.labels:
            label = torch.tensor((float(self.annotations.iloc[index, 2])))
            return img, label, image_id
        else:
            return img, image_id
