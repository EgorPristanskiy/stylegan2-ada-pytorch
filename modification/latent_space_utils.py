import numpy as np

class LatentSpaceLoader:
    @staticmethod
    def load_latent_space(path_to_archive: str) -> np.ndarray:
        data = np.load(path_to_archive)
        return data.get('w')
