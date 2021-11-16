import numpy as np
from enum import Enum

class Noise_Enum(Enum):
    WHITE="white_noise"
    PINK="pink_noise"
    BROWN="brownian_noise"
    QUANT="quantization"

class Noise:
    type                = Noise_Enum.WHITE
    parameters          = []
    seed                = 0

    def __init__(self,type,parameters,seed = 0):
        self.type               = type
        self.parameters         = parameters
        self.seed               = seed

    def apply_noise(self,y):
        match self.type:
            case Noise_Enum.WHITE:
                return self.__apply_noise_white(y)
            case Noise_Enum.PINK:
                return self.__apply_noise_pink(y)
            case Noise_Enum.BROWN:
                return self.__apply_noise_brown(y)
            case Noise_Enum.QUANT:
                return self.__apply_noise_quant(y)

    # Ab hier editieren

    def __apply_noise_white(self,y):
        np.random.seed(self.seed)
        return y + np.random.normal(0, self.parameters, len(y))

    def __apply_noise_pink(self,y):
        return y

    def __apply_noise_brown(self,y):
        return y

    def __apply_noise_quant(self,y):
        return y
