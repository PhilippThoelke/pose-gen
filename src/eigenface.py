import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import Subset
from torchvision.datasets import LFWPeople, Flowers102
from torchvision.transforms import Compose, ToTensor, Lambda, Resize
from models import Eigenface


if __name__ == "__main__":
    num_images = 2500
    resize = True
    resize_shape = (160, 160)

    data_transforms = ToTensor()
    if resize:
        data_transforms = Compose([data_transforms, Resize(resize_shape)])
    data = LFWPeople("data/", transform=data_transforms, download=True)
    data = Subset(data, torch.randperm(len(data))[:num_images])
    x = np.stack(
        [x.permute(1, 2, 0).numpy() for x, _ in tqdm(data, desc="loading data")], axis=0
    )

    print("Running PCA...", end="")
    model = Eigenface(x)
    print("done")
    with open("models/eigenface.pkl", "wb") as f:
        pickle.dump(model, f)
