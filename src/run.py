import cv2
import pickle
import numpy as np
from queue import deque
from models import Eigenface, VAE
from latent import HandLatent


def main(
    model,
    example_raw=None,
    img_size=(256, 256),
    latent_scaling=1,
    latent_vel_factor=0.005,
    latent_damping=0.8,
    dynamic_latent=False,
    adaptive_scaling=True,
    latent_buffer_size=1000,
):
    # set up camera capture and pose model
    cap = cv2.VideoCapture(0)
    latent_generator = HandLatent(model.latent_dim)

    # initialize latent vector
    latent, latent_vel, previous_z = np.zeros((3, model.latent_dim), dtype=np.float32)

    # landmark running averages
    if example_raw is not None:
        example_raw = example_raw.reshape(example_raw.shape[0], -1)
        initial_buffer = latent_generator.transform(example_raw)
        landmark_buffer = deque(initial_buffer, maxlen=latent_buffer_size)
    else:
        landmark_buffer = deque(maxlen=latent_buffer_size)

    cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        # read a frame from the camera
        _, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # extract hand landmarks
        z = latent_generator.get_latent(img_rgb)
        if z is not None:
            # normalize landmark points individually
            if adaptive_scaling:
                landmark_buffer.append(z)
            if len(landmark_buffer) > 1:
                landmark_buffer_arr = np.asarray(landmark_buffer)
                mean = landmark_buffer_arr.mean(axis=0)
                std = landmark_buffer_arr.std(axis=0)
                z = (z - mean) / std
            # scale latent dim according to network
            z = z * latent_scaling
            z = z * model.latent_std + model.latent_mean
            if dynamic_latent:
                # update latent vector
                latent_vel *= latent_damping
                magnitude = np.linalg.norm(previous_z - z)
                latent_vel += (z - latent) * magnitude * latent_vel_factor
                latent += latent_vel
            else:
                latent = z
            # generate image
            img = model.generate(latent)
            # visualize the image
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, img_size)
            cv2.imshow("img", img)
            cv2.waitKey(1)


if __name__ == "__main__":
    # eigenface or VAE
    model_type = "eigenface"

    # load example data to generate latent mean and std
    data = np.load("data/hands.npy")

    # load model
    if model_type == "VAE":
        model = VAE.load_from_checkpoint(
            "models/lightning_logs/version_1/checkpoints/epoch=1-step=3310.ckpt"
        )
        model.eval().freeze()
    elif model_type == "eigenface":
        with open("models/eigenface.pkl", "rb") as f:
            model = pickle.load(f)
    else:
        raise ValueError(f"Unknown model type {model_type}")

    main(model, example_raw=data)
