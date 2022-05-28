import torch
import numpy as np
from queue import deque
from train import VAE
import cv2
import mediapipe as mp


def main(latent_scaling=2, latent_vel_factor=0.01, latent_damping=0.8):
    # load data
    data = np.load("data/hands.npy")

    # load model
    model = VAE.load_from_checkpoint(
        "lightning_logs/version_11/checkpoints/epoch=0-step=1655.ckpt"
    )
    model.eval().freeze()

    # landmark projection layer
    landmark_proj = torch.empty(data[0].size, model.latent_dim)
    torch.nn.init.xavier_normal_(landmark_proj)

    # set up camera capture and pose model
    cap = cv2.VideoCapture(0)
    detector = mp.solutions.hands.Hands(max_num_hands=1)

    # initialize latent vector
    latent, latent_vel, previous_z = torch.zeros(3, model.latent_dim)

    # landmark running averages
    initial_buffer = data.reshape(data.shape[0], -1) @ landmark_proj.numpy()
    landmark_buffer = deque(initial_buffer, maxlen=1000)

    while True:
        # read a frame from the camera
        _, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # extract hand landmarks
        res = detector.process(img_rgb)
        if res.multi_hand_world_landmarks is not None:
            landmarks = torch.tensor(
                [
                    [pt.x, pt.y, pt.z]
                    for pt in res.multi_hand_world_landmarks[0].landmark
                ]
            )
            # hand landmarks to latent distribution
            z = landmarks.flatten() @ landmark_proj
            # normalize landmark points individually
            landmark_buffer.append(z.numpy())
            if len(landmark_buffer) > 1:
                landmark_buffer_arr = np.array(landmark_buffer)
                mean = landmark_buffer_arr.mean(axis=0)
                std = landmark_buffer_arr.std(axis=0)
                z = (z - mean) / std
            # scale latent dim according to network
            z = z * latent_scaling
            z = z * model.running_std + model.running_mean
            # update latent vector
            latent_vel *= latent_damping
            magnitude = (previous_z - z).norm()
            latent_vel += (z - latent) * magnitude * latent_vel_factor
            latent += latent_vel
            # generate image
            img = model.decode(latent).permute(1, 2, 0).numpy() / 2 + 0.5
            # visualize the image
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, [512, 512])
            cv2.imshow("img", img)
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
