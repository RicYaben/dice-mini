import warnings
from importlib.metadata import version

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

__version__ = version("dice-mini")
