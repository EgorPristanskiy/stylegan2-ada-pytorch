from .latent_space_utils import LatentSpaceLoader as Loader

class LatentSpaceModifier:
    def __init__(
        self,
        lowest_param_space_file  : str,
        highest_param_space_file : str,
        target_latent_space      : str,
        modification_shift       : float
    ):
        lowest_value_space  = Loader.load_latent_space(lowest_param_space_file)
        highest_value_space = Loader.load_latent_space(highest_param_space_file)

        self._delta_space         = highest_value_space - lowest_value_space
        self._target_latent_space = Loader.load_latent_space(target_latent_space)
        self._modification_shift  = modification_shift
    
    def _modify_target_space(self):
        self._target_latent_space += self._delta_space * self._modification_shift
    
    def move_image(self):
        self._modify_target_space()
        return self._target_latent_space
