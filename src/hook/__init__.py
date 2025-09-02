# Only import the new fusion function to avoid torch dependency issues
try:
    from .train_fusionmmae import train_fusionmmae
    from .train_fusionmmae_multi_learner import train_fusionmmae_multi_learner

    __all__ = ["train_fusionmmae", "train_fusionmmae_multi_learner"]
except ImportError:
    __all__ = []
