from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, Flowers102, LFWPeople
from torchvision.transforms import Compose, ToTensor, Lambda, Resize
import pytorch_lightning as pl
from models import VAE


if __name__ == "__main__":
    # load data
    data_transforms = Compose(
        [ToTensor(), Lambda(lambda x: x * 2 - 1), Resize((128, 128))]
    )
    data = LFWPeople("data/", transform=data_transforms, download=True)
    dl = DataLoader(data, batch_size=8, shuffle=True, num_workers=4)

    # create lightning module
    module = VAE(channels=data[0][0].size(0))

    # train model
    trainer = pl.Trainer(
        max_epochs=-1,
        log_every_n_steps=32,
        accelerator="auto",
        devices="auto",
        default_root_dir="models/",
    )
    trainer.fit(module, dl)
