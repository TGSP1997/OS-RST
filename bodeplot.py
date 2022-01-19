from re import S
from matplotlib.pyplot import show
from matplotlib import ticker, cm
from scipy.optimize import minimize
from scipy.fft import fft, ifft, fftfreq
from scipy import signal
from numpy import ma

from input_function import *
from noises import *
from filter import *
from plot_sig import *
from cost import *

kalman = Filter(Filter_Enum.KALMAN, [])
pt1 = Filter(Filter_Enum.PT1, [])

window_len_white = 14
N = window_len_white//2
noise_std_dev = 0.1

# Übertragungsfunktion des Filters bestimmen
point_counter = 500

u = np.zeros(int(point_counter))
u[point_counter//2] = 1
t = np.linspace(0,1,num = int(point_counter))

y1      = kalman.filter_fun(t,u,para = [2,[0,0],noise_std_dev, 1e4])[0]
y2      = pt1.filter_fun(t, u, para = 100)

# wiener
h_opt_noncausal = [0.0496811771626988, 0.06947908478423444, 0.011569497877491324, 0.12133430261881151, 0.13475784870233373, 0.06456927694523334, 0.05501950128199152, 0.06456927694523343, 0.13475784870233454, 0.12133430261881255, 0.011569497877490866, 0.069479084784235, 0.04968117716269913]
y_start_pad = u[0]*np.ones(N)
y_end_pad = u[-1]*np.ones(N)
y3 = np.convolve(np.append(y_start_pad, np.append(u, y_end_pad)), h_opt_noncausal, mode='full')
y3 = y3[2*N-1:-2*N+1]

plot_bode = Plot_Sig(Plot_Enum.BODE,"Bode Plot",parameters = 0)

plot_bode.plot_sig(t,[[u],[y3]],[
        "$\sigma$ = 0.1"])

plt.show()