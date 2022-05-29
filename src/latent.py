import numpy as np
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
            # initialize a random projection matrix with a uniform xavier distribution
            xavier_range = np.sqrt(6) / np.sqrt(
                HandLatent.LANDMARK_DIMENSION + dimension
            )
            self.projection = np.random.uniform(
                -xavier_range,
                xavier_range,
                size=(HandLatent.LANDMARK_DIMENSION, dimension),
            ).astype(np.float32)

    def get_latent(self, img):
        res = self.detector.process(img)

        if res.multi_hand_world_landmarks is None:
            return None

        landmarks = np.array(
            [[pt.x, pt.y, pt.z] for pt in res.multi_hand_world_landmarks[0].landmark]
        )
        return self.transform(landmarks.ravel())

    def transform(self, x):
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)
        if self.projection is None:
            return x
        return x @ self.projection
