import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, Flowers102, LFWPeople
from torchvision.transforms import Compose, ToTensor, Lambda, Resize
import pytorch_lightning as pl


class VAE(pl.LightningModule):
    def __init__(
        self,
        channels=3,
        h_dims=[8, 16, 32, 64],
        latent_dim=1024,
        latent_img_size=2,
        kl_weight=2.5e-4,
        learning_rate=5e-3,
        alpha=0.99,
        warmup_steps=300,
        std_clip=2,
    ):
        super().__init__()
        self.channels = channels
        self.h_dims = h_dims
        self.latent_dim = latent_dim
        self.latent_img_size = latent_img_size
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.std_clip = torch.scalar_tensor(std_clip)

        h_dims.insert(0, channels)

        # encoder
        modules = []
        for i in range(1, len(h_dims)):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        h_dims[i - 1],
                        out_channels=h_dims[i],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        padding_mode="reflect",
                    ),
                    nn.BatchNorm2d(h_dims[i]),
                    nn.LeakyReLU(),
                )
            )
        self.encoder = nn.Sequential(*modules)

        # parametrization layers
        self.alpha = alpha
        self.parametrization = nn.Linear(
            h_dims[-1] * latent_img_size * latent_img_size, latent_dim * 2
        )
        self.decoder_in = nn.Linear(
            latent_dim, h_dims[-1] * latent_img_size * latent_img_size
        )

        # decoder
        modules = []
        for i in range(2, len(h_dims)):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        h_dims[-i + 1],
                        h_dims[-i],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    *(
                        [nn.Tanh()]
                        if i == len(h_dims)
                        else [nn.BatchNorm2d(h_dims[-i]), nn.LeakyReLU()]
                    )
                )
            )
        self.decoder = nn.Sequential(*modules)
        self.outnet = nn.Sequential(
            ResidualBlock(h_dims[1], h_dims[1]),
            ResidualBlock(h_dims[1], channels, out=True),
        )

        # store the image size
        self.register_buffer("img_size", torch.tensor([-1, -1]))
        self.register_buffer("encoded_img_size", torch.tensor([-1, -1]))

        # store running averages of latent mean and std
        self.register_buffer("running_mean", torch.zeros(latent_dim))
        self.register_buffer("running_std", torch.zeros(latent_dim))

    def encode(self, x):
        self.img_size = torch.tensor(x.shape[-2:])
        x = self.encoder(x)
        self.encoded_img_size = torch.tensor(x.shape[-2:])
        x = F.adaptive_max_pool2d(x, (self.latent_img_size, self.latent_img_size))
        x = x.flatten(start_dim=1)
        mu, logvar = self.parametrization(x).split(self.latent_dim, dim=1)
        logvar = logvar.clip(max=2 * self.std_clip.log())
        return mu, logvar

    def decode(self, x):
        has_batch = x.ndim > 1
        if not has_batch:
            x = x.unsqueeze(0)
        x = self.decoder_in(x)
        x = x.view(
            x.size(0), self.h_dims[-1], self.latent_img_size, self.latent_img_size
        )
        x = F.interpolate(x, self.encoded_img_size.tolist(), mode="bilinear")
        x = self.decoder(x)
        x = F.interpolate(x, self.img_size.tolist(), mode="nearest")
        x = self.outnet(x)
        if has_batch:
            return x
        return x[0]

    def reparametrize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return torch.randn_like(mu) * std + mu

    def forward(self, x, return_parametrization=False):
        mu, logvar = self.encode(x)
        x = self.reparametrize(mu, logvar)
        x = self.decode(x)
        if return_parametrization:
            return x, mu, logvar
        return x

    def training_step(self, batch):
        x = batch[0]
        recons, mu, logvar = self(x, return_parametrization=True)
        # reconstruction loss
        recons_loss = F.mse_loss(recons, x)
        # KL divergence loss
        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0
        )
        # total loss
        loss = recons_loss + self.kl_weight * kl_loss

        # update running averages
        std = (0.5 * logvar).exp()
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * mu.mean(
            dim=0
        )
        self.running_std = self.alpha * self.running_std + (1 - self.alpha) * std.mean(
            dim=0
        )

        # logging
        self.log("loss", loss)
        self.log("recons_loss", recons_loss)
        self.log("kl_loss", kl_loss)
        self.log("lr", self.optimizers().param_groups[0]["lr"])
        self.logger.experiment.add_histogram("mu", mu, global_step=self.global_step)
        self.logger.experiment.add_histogram("std", std, global_step=self.global_step)
        self.logger.experiment.add_images("input", x / 2 + 0.5)
        self.logger.experiment.add_images("reconstruction", recons / 2 + 0.5)
        return loss

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.learning_rate)
        opt.param_groups[0]["initial_lr"] = self.learning_rate
        sched = ExponentialLR(opt, gamma=0.7, last_epoch=-1)
        return dict(optimizer=opt, lr_scheduler=sched)

    def optimizer_step(self, *args, **kwargs):
        if self.global_step < self.warmup_steps:
            self.optimizers().param_groups[0]["lr"] = self.learning_rate * (
                (self.global_step + 1) / self.warmup_steps
            )
        return super().optimizer_step(*args, **kwargs)


class ResidualBlock(nn.Module):
    def __init__(self, d_in, d_out, out=False):
        super().__init__()
        self.out = out
        self.conv = nn.Sequential(
            nn.Conv2d(d_in, d_out * 2, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(d_out * 2),
            nn.LeakyReLU(),
            nn.Conv2d(d_out * 2, d_out, 3, padding=1, padding_mode="reflect"),
        )
        self.residual = nn.Conv2d(d_in, d_out, 1)
        if not out:
            self.norm = nn.BatchNorm2d(d_out)

    def forward(self, x):
        x = self.conv(x) + self.residual(x)
        if self.out:
            return torch.tanh(x)
        return F.leaky_relu(self.norm(x))


if __name__ == "__main__":
    # load data
    data_transforms = Compose(
        [ToTensor(), Lambda(lambda x: x * 2 - 1), Resize((200, 200))]
    )
    data = LFWPeople("data/", transform=data_transforms)
    dl = DataLoader(data, batch_size=8, shuffle=True, num_workers=4)

    # create lightning module
    module = VAE(channels=data[0][0].size(0))

    # train model
    trainer = pl.Trainer(
        max_epochs=-1, log_every_n_steps=32, accelerator="auto", devices="auto"
    )
    trainer.fit(module, dl)
