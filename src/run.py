import time
from glob import glob
from queue import deque

import cv2
import numpy as np
import torch

from latent import FaceLatent, HandLatent, OSCLatent, SoundLatent
from model import DCGAN, PGAN, VAE, Eigenface


def main(
    img_model,
    latent_model,
    example_raw=None,
    img_size=(512, 512),
    latent_scaling=.5,
    latent_damping=True,
    latent_vel_factor=0.01,
    latent_damping_factor=0.85,
    undamped_latent_scale=0.15,
    adaptive_scaling=True,
    latent_buffer_size=1000,
    fullscreen=True,
):
    # initialize latent vector
    latent, latent_vel, previous_z = torch.zeros((3, img_model.latent_dim))

    # landmark running averages
    if example_raw is not None:
        example_raw = example_raw.reshape(example_raw.shape[0], -1)
        initial_buffer = latent_model.transform(example_raw)
        landmark_buffer = deque(initial_buffer, maxlen=latent_buffer_size)
    else:
        landmark_buffer = deque(maxlen=latent_buffer_size)

    if fullscreen:
        cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    last_frame = time.time()
    while True:
        # compute latent vector
        z = latent_model.get_latent()
        if z is not None:
            # normalize latent variables individually
            if adaptive_scaling:
                landmark_buffer.append(z)
            if len(landmark_buffer) > 1:
                landmark_buffer_arr = torch.stack(list(landmark_buffer))
                mean = landmark_buffer_arr.mean(dim=0)
                std = landmark_buffer_arr.std(dim=0)
                z = (z - mean) / std
            # scale latent dim according to network
            z = z * latent_scaling
            z = z * img_model.latent_std + img_model.latent_mean
            if latent_damping:
                # update latent vector
                latent_vel *= latent_damping_factor
                magnitude = (previous_z - z).norm()
                latent_vel += (z - latent) * magnitude * latent_vel_factor
                latent += latent_vel
            # generate image
            img = img_model.generate(
                latent + z * (undamped_latent_scale if latent_damping else 1)
            ).numpy()
            # visualize the image
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, img_size)
            cv2.imshow("img", img)
            cv2.waitKey(1)

        curr_frame = time.time()
        print(f"{1 / (curr_frame - last_frame):.2f} FPS")
        last_frame = curr_frame


if __name__ == "__main__":
    # eigenface, kernel-eigenface, DCGAN, PGAN or VAE
    img_model_type = "PGAN"
    # hand, face, sound or OSC
    latent_model_type = "OSC"
    # whether to jit compile the image model
    jit_img_model = True

    # load model
    if img_model_type == "VAE":
        path = glob("models/lightning_logs/version_7/checkpoints/*.ckpt")[0]
        img_model = VAE.load_from_checkpoint(path)
        img_model.eval().freeze()
    elif img_model_type == "eigenface":
        img_model = torch.load("models/eigenface.pt")
    elif img_model_type == "kernel-eigenface":
        img_model = torch.load("models/kernel-eigenface.pt")
    elif img_model_type == "DCGAN":
        img_model = DCGAN("models/gan4-epoch110.model")
    elif img_model_type == "PGAN":
        img_model = PGAN.load_checkpoint(pretrained=True, model_name="celebAHQ-512")
    else:
        raise ValueError(f"Unknown model type {img_model_type}")

    if jit_img_model:
        try:
            m = torch.jit.trace(img_model, torch.randn(img_model.latent_dim))
            m.latent_dim = img_model.latent_dim
            m.latent_mean = img_model.latent_mean
            m.latent_std = img_model.latent_std
            m.generate = img_model.generate
            img_model = m
        except:
            print("Failed to JIT trace the model. Continuing...")

    # create latent model and potentially load example data
    data = None
    if latent_model_type == "hand":
        latent_model = HandLatent(img_model.latent_dim)
        # data = np.load("data/hands.npy")
    elif latent_model_type == "face":
        latent_model = FaceLatent(img_model.latent_dim)
    elif latent_model_type == "sound":
        latent_model = SoundLatent(img_model.latent_dim)
    elif latent_model_type == "OSC":
        latent_model = OSCLatent(img_model.latent_dim)
    else:
        raise ValueError(f"Unknown latent type {latent_model_type}")

    main(img_model, latent_model, example_raw=data)
