"""Models for cluster research."""

from importlib.metadata import version, PackageNotFoundError
from .model import CTEN
from .utils import *


try:
    __version__ = version("iccas-cluster")
except PackageNotFoundError:
    # package is not installed
    pass
