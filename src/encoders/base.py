from abc import ABC, abstractmethod
from ast import Not
import numpy as np
import gym


class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, image):
        pass

    @abstractmethod
    def decode(self, image):
        pass

    @abstractmethod
    def update(self, batch_of_images):
        pass
