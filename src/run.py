import cv2
import torch
import numpy as np
from queue import deque
from models import Eigenface, VAE
from latent import HandLatent, FaceLatent


def main(
    img_model,
    latent_model,
    example_raw=None,
    img_size=(512, 512),
    latent_scaling=1,
    latent_vel_factor=0.001,
    latent_damping=0.7,
    dynamic_latent=False,
    adaptive_scaling=True,
    latent_buffer_size=1000,
    fullscreen=True,
):
    # set up camera capture
    cap = cv2.VideoCapture(0)

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

    while True:
        # read a frame from the camera
        _, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # extract hand landmarks
        z = latent_model.get_latent(img_rgb)
        if z is not None:
            # normalize landmark points individually
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
            if dynamic_latent:
                # update latent vector
                latent_vel *= latent_damping
                magnitude = (previous_z - z).norm()
                latent_vel += (z - latent) * magnitude * latent_vel_factor
                latent += latent_vel
            else:
                latent = z
            # generate image
            img = img_model.generate(latent).numpy()
            # visualize the image
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, img_size)
            cv2.imshow("img", img)
            cv2.waitKey(1)


if __name__ == "__main__":
    # eigenface or VAE
    img_model_type = "eigenface"
    # hand or face
    latent_model_type = "hand"

    # load model
    if img_model_type == "VAE":
        img_model = VAE.load_from_checkpoint(
            "models/lightning_logs/version_1/checkpoints/epoch=1-step=3310.ckpt"
        )
        img_model.eval().freeze()
    elif img_model_type == "eigenface":
        img_model = torch.load("models/eigenface.pt")
    else:
        raise ValueError(f"Unknown model type {img_model_type}")

    # create latent model and potentially load example data
    data = None
    if latent_model_type == "hand":
        latent_model = HandLatent(img_model.latent_dim)
        data = np.load("data/hands.npy")
    elif latent_model_type == "face":
        latent_model = FaceLatent(img_model.latent_dim)
    else:
        raise ValueError(f"Unknown latent type {latent_model_type}")

    main(img_model, latent_model, example_raw=data)
