import torch
import mediapipe as mp


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
            return x
        return x @ self.projection


class HandLatent(LatentProvider):
    def __init__(self, dimension, random_projection=True):
        super().__init__(dimension, random_projection, 63)
        self.detector = mp.solutions.hands.Hands(max_num_hands=1)

    def get_latent(self, img):
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

    def get_latent(self, img):
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
