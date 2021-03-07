import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
#import nori2 as nori

"""
Define a base dataset contains some function I always useself.
"""


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError


    def transform_initialize(self, crop_size, config=['random_crop', 'to_tensor', 'norm']):
        """
        Initialize the transformation oprs and create transform function for img
        """
        self.transforms_oprs = {}
        self.transforms_oprs["hflip"]= transforms.RandomHorizontalFlip(0.5)
        self.transforms_oprs["vflip"] = transforms.RandomVerticalFlip(0.5)
        self.transforms_oprs["random_crop"] = transforms.RandomCrop(crop_size)
        self.transforms_oprs["to_tensor"] = transforms.ToTensor()
        self.transforms_oprs["norm"] = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.transforms_oprs["resize"] = transforms.Resize(crop_size)
        self.transforms_oprs["center_crop"] = transforms.CenterCrop(crop_size)
        self.transforms_oprs["rdresizecrop"] = transforms.RandomResizedCrop(crop_size, scale=(0.7, 1.0), ratio=(1,1), interpolation=2)
        self.transforms_fun = transforms.Compose([self.transforms_oprs[name] for name in config])

    def loader(self, **args):
        return DataLoader(dataset=self, **args)

    @staticmethod
    def read_img(path):
        """
        Read Images
        """
        img = Image.open(path)

        return img
