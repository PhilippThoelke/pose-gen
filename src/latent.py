import cv2
import torch
import mediapipe as mp
from queue import deque
import sounddevice as sd
from scipy.ndimage import gaussian_filter1d


class LatentProvider:
    def __init__(self, dimension, random_projection, input_dimension=None):
        self.dimension = dimension

        self.projection = None
        if random_projection:
            self.projection = torch.empty(input_dimension, dimension)
            torch.nn.init.xavier_uniform_(self.projection)

    def get_latent(self, *args):
        raise NotImplementedError()

    def transform(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if self.projection is None:
            return x[: self.dimension]
        return x @ self.projection


class HandLatent(LatentProvider):
    def __init__(self, dimension, random_projection=True):
        super().__init__(dimension, random_projection, 63)
        self.detector = mp.solutions.hands.Hands(max_num_hands=1)

        # set up camera capture
        self.cap = cv2.VideoCapture(0)

    def get_latent(self):
        # read a frame from the camera
        _, img = self.cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # detect hand landmarks
        res = self.detector.process(img)

        if res.multi_hand_world_landmarks is None:
            return None

        landmarks = torch.tensor(
            [[pt.x, pt.y, pt.z] for pt in res.multi_hand_world_landmarks[0].landmark]
        )
        return self.transform(landmarks.flatten())


class FaceLatent(LatentProvider):
    def __init__(self, dimension, random_projection=True):
        super().__init__(dimension, random_projection, 1404)
        self.detector = mp.solutions.face_mesh.FaceMesh()

        # set up camera capture
        self.cap = cv2.VideoCapture(0)

    def get_latent(self):
        # read a frame from the camera
        _, img = self.cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # detect face landmarks
        res = self.detector.process(img)

        if res.multi_face_landmarks is None:
            return None

        landmarks = torch.tensor(
            [[pt.x, pt.y, pt.z] for pt in res.multi_face_landmarks[0].landmark]
        )

        mean = landmarks.mean(dim=0, keepdims=True)
        std = landmarks.std(dim=0, keepdims=True)
        landmarks = (landmarks - mean) / std
        return self.transform(landmarks.flatten())


class SoundLatent(LatentProvider):

    BUFFER_SIZE = 500
    SPECTRUM_SIZE = 500
    DOWNSAMPLE = 10
    BLUR_STRENGTH = 5

    def __init__(self, dimension, random_projection=True):
        super().__init__(dimension, random_projection, SoundLatent.SPECTRUM_SIZE)
        self.stream = sd.InputStream(channels=1, callback=self.audio_callback)
        self.stream.start()
        self.audio_buffer = deque(maxlen=SoundLatent.BUFFER_SIZE)

    def get_latent(self):
        signal = torch.tensor(self.audio_buffer)
        spec = torch.fft.fft(signal, n=SoundLatent.SPECTRUM_SIZE * 2).abs()
        latent = spec[: SoundLatent.SPECTRUM_SIZE]
        latent = gaussian_filter1d(latent, SoundLatent.BLUR_STRENGTH)
        return self.transform(latent.flatten())

    def audio_callback(self, indata, *args, **kwargs):
        self.audio_buffer.extend(indata[:: SoundLatent.DOWNSAMPLE, 0])
