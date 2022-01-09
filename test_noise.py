from cost import Cost_Enum
from scipy.optimize import minimize
from scipy import signal

from input_function import *
from noises import *
from filter import *
from plot_sig import *
from cost import *

noise_std_dev = 0.1
sine    = Input_Function(Input_Enum.SINE, [1, 1, 0, 0], point_counter=1e3)
polynom = Input_Function(Input_Enum.POLYNOM, [1,1,1,1]) #coefs in descending order 2x^2+1 = [2,0,1]
exp     = Input_Function(Input_Enum.EXP, [1,2,0,0]) #coefs [a,b,c,d]= a*e^(t/b+c)+d
whites  = [Noise(Noise_Enum.WHITE, noise_std_dev, seed=i) for i in range(10)]
pinks   = [Noise(Noise_Enum.PINK, noise_std_dev, seed=i) for i in range(10)]
browns   = [Noise(Noise_Enum.BROWN, noise_std_dev, seed=i) for i in range(10)]
quant   = Noise(Noise_Enum.QUANT, 0.5)
plot    = Plot_Sig(Plot_Enum.MULTI, "Overview", [])

cost    = Cost(Cost_Enum.PHASE_SHIFT)

time, true_sine, true_sine_dot = sine.get_fun()
norm_freq = time[:round(len(time))] / (time[-1] - time[0])

print(cost.cost(true_sine_dot, true_sine))

y_white = whites[0].apply_noise(true_sine)
y = browns[0].apply_noise(true_sine)
y2 = pinks[0].apply_noise(true_sine)

### plot power spectral density
'''
freqs, psd = signal.welch(y)
freqs, psd2 = signal.welch(y2)
psd = 20*np.log10(psd)
psd2 = 20*np.log10(psd2)

plt.figure(figsize=(5, 4))
plt.semilogx(freqs, psd, freqs, psd2)
plt.title('PSD: power spectral density')
plt.xlabel('Frequency')
plt.ylabel('dB')
plt.tight_layout()
'''

corr = np.correlate(y_white - true_sine, y_white - true_sine, mode='full')
corr = 1/1000 * corr[round(corr.size/2)-1:]
plot.plot_sig(time, [y_white - true_sine, corr],["Roh", "R_nn"])
plt.show()