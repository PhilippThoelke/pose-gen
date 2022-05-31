import torch
import mediapipe as mp


class LatentProvider:
    def __init__(self, dimension):
        self.dimension = dimension

    def get_latent(self, *args):
        raise NotImplementedError()

    def transform(self, x):
        raise NotImplementedError()


class HandLatent(LatentProvider):

    LANDMARK_DIMENSION = 63

    def __init__(self, dimension, random_projection=True):
        super().__init__(dimension)
        self.detector = mp.solutions.hands.Hands(max_num_hands=1)

        self.projection = None
        if random_projection:
            self.projection = torch.empty(HandLatent.LANDMARK_DIMENSION, dimension)
            torch.nn.init.xavier_uniform_(self.projection)

    def get_latent(self, img):
        res = self.detector.process(img)

        if res.multi_hand_world_landmarks is None:
            return None

        landmarks = torch.tensor(
            [[pt.x, pt.y, pt.z] for pt in res.multi_hand_world_landmarks[0].landmark]
        )
        return self.transform(landmarks.flatten())

    def transform(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if self.projection is None:
            return x
        return x @ self.projection
