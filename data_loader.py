from torchvision import datasets, transforms
import torch
import numpy as np
import os
def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()
         ])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):

    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    # print(list(data.imgs))
    names = list(map(lambda x: os.path.basename(x[0]), list(data.imgs)))
    label = list(map(lambda x: x[1], list(data.imgs)))
    # print(names, label)
    # for name, label in data.imgs:
    #     print(name, label)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader, names, label
