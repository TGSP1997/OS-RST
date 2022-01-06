from cost import Cost_Enum
from scipy.optimize import minimize

from input_function import *
from noises import *
from filter import *
from plot_sig import *
from cost import *

noise_std_dev = 0.1

sine    = Input_Function(Input_Enum.SINE, [1, 0.1, 0, 0.5])
white   = Noise(Noise_Enum.WHITE, noise_std_dev)
pink   = Noise(Noise_Enum.PINK, noise_std_dev)
brown   = Noise(Noise_Enum.BROWN, noise_std_dev)
quant   = Noise(Noise_Enum.QUANT, 0.5)
plot    = Plot_Sig(Plot_Enum.MULTI, "Overview", [])

time, true_sine, true_sine_dot = sine.get_fun()
y = quant.apply_noise(true_sine)

plot.plot_sig(time, [true_sine, y],["Roh","Rausch"])
plt.show()