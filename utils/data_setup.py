"""
data_setup.py, contains functionality for creating PyTorch Dataloaders for 
image classification. 
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()


def create_dataloaders(train_dir: str,
                       test_dir: str,
                       transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int = NUM_WORKERS):

    # Use torchvision.datasets.ImageFoler() to store images and their corresponding samples, https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html.
    # Note: Images must be arraged as follows, root/class_name/image.png, where directories correspond to class labels.
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Convert images into dataloaders, torch.utils.data.DataLoader(), https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
    # DataLoader provides an iterable around ImageFolder to enable easy access to samples of our dataset.
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,  # shuffles data after every epoch
                                  num_workers=num_workers,  # subprocesses to use for dataloading
                                  pin_memory=True)  # copies tensors into device/CUDA pinned memory before returning them
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False,  # only set to True during training
                                 num_workers=num_workers,
                                 pin_memory=True)

    return train_dataloader, test_dataloader, class_names
