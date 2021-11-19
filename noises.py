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
        """Generates pink noise using the Voss-McCartney algorithm (source: https://www.dsprelated.com/showarticle/908.php)
    
        rcols: number of random sources to add
        
        returns: signal with added noise as np array
        """
        np.random.seed(self.seed)
        nrows = len(y)
        array = np.empty((nrows, ncols))
        array.fill(np.nan)
        array[0, :] = np.random.random(ncols)
        array[:, 0] = np.random.random(nrows)
        
        # the total number of changes is nrows
        n = nrows
        cols = np.random.geometric(0.5, n)
        cols[cols >= ncols] = 0
        rows = np.random.randint(nrows, size=n)
        array[rows, cols] = np.random.random(n)

        df = pd.DataFrame(array)
        df.fillna(method='ffill', axis=0, inplace=True)
        total = df.sum(axis=1)
        y = y + total.values
        return y

    def __apply_noise_brown(self,y):
        """
        Brown noise (source: https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/generator.py)
        :type state: :class:`np.random.RandomState`
        Power decreases with -3 dB per octave.
        Power density decreases with 6 dB per octave.
        """
        np.random.seed(self.seed)
        N = len(y)
        state = np.random.RandomState()
        uneven = N % 2
        X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
        S = (np.arange(len(X)) + 1)  # Filter
        noise = (irfft(X / S)).real
        if uneven:
            noise = noise[:-1]
        y = y + normalize(noise)
        return y

    def __apply_noise_quant(self,y):

        return y
