import os

import torch
from torch.utils.data import Subset
from torchvision.datasets import EuroSAT, Flowers102, LFWPeople
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

from data import MRI
from model import Eigenface, KernelEigenface


if __name__ == "__main__":
    num_images = 5000
    resize = True
    resize_shape = (512, 512)
    use_kernel_pca = False
    latent_dim = 200

    data_transforms = ToTensor()
    if resize:
        data_transforms = Compose([data_transforms, Resize(resize_shape)])

    # define the dataset
    data = LFWPeople("~/data", transform=data_transforms, download=True)
    # data = Flowers102("data/", transform=data_transforms, download=True)
    # data = EuroSAT("data/", transform=data_transforms, download=True)
    # data = MRI("data/", transform=data_transforms)

    data = Subset(data, torch.randperm(len(data))[:num_images])
    x = torch.stack(
        [x.permute(1, 2, 0) for x, _ in tqdm(data, desc="loading data")], axis=0
    )

    if not os.path.exists("models/"):
        os.makedirs("models/")

    if not use_kernel_pca:
        print("Running PCA...", end="")
        model = Eigenface(x, latent_dim=latent_dim)
        print("done")
        torch.save(model, "models/eigenface.pt")
    else:
        print("Running Kernel PCA...", end="")
        model = KernelEigenface(x, latent_dim=latent_dim)
        print("done")
        torch.save(model, "models/kernel-eigenface.pt")
