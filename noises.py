import numpy as np
from enum import Enum
import colorednoise as cn

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
        """Generates and applies pink noise based on Timmer, J. and Koenig, M.: On generating power law noise. Astron. Astrophys. 300, 707-710 (1995) 
        (source: https://github.com/felixpatzelt/colorednoise)
        
        std_dev: standard deviation of noise

        returns: signal with added noise as np array
        """
        np.random.seed(self.seed)
        beta = 1 # the exponent
        samples = len(y) # number of samples to generate
        std_dev = self.parameters 
        y = y + std_dev * cn.powerlaw_psd_gaussian(beta, samples)
        return y

    def __apply_noise_brown(self,y):
        """
        Generates and applies brown noise based on Timmer, J. and Koenig, M.: On generating power law noise. Astron. Astrophys. 300, 707-710 (1995) 
        (source: https://github.com/felixpatzelt/colorednoise)

        std_dev: standard deviation of noise

        returns: signal with added noise as np array        
        """
        np.random.seed(self.seed)
        beta = 2 # the exponent
        samples = len(y) # number of samples to generate
        std_dev = self.parameters 
        y = y + std_dev * cn.powerlaw_psd_gaussian(beta, samples)
        return y

    def __apply_noise_quant(self,y):
        """
        Generates and applies quantization noise

        resolution: resolution of the measurement system

        returns: signal with added noise as np array
        """
        resolution = self.parameters
        return (resolution * np.around(y/resolution))
