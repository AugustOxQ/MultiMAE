# Only import the new fusion function to avoid torch dependency issues
try:
    from .train_fusionmmae import train_fusionmmae

    __all__ = ["train_fusionmmae"]
except ImportError:
    __all__ = []
