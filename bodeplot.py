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

noise_std_dev = 0.1

# Übertragungsfunktion des Filters bestimmen
point_counter = 500

u = np.zeros(int(point_counter))
u[point_counter//2] = 1
t = np.linspace(0,1,num = int(point_counter))

y1      = kalman.filter_fun(t,u,para = [2,[0,0],noise_std_dev, 1e4])[0]
y2      = pt1.filter_fun(t, u, para = 100)

# wiener
h_wiener = np.array([    [0.08110043, 0.09482883, 0.02395821, 0.12631018, 0.12790999, 0.03793559, 0.01077581, 0.03793559, 0.12790999, 0.12631018, 0.02395821, 0.09482883, 0.08110043],
                         [0.06005379, 0.0781426 , 0.01643154, 0.12433697, 0.13439967, 0.05805149, 0.04271674, 0.05805149, 0.13439967, 0.12433697, 0.01643154, 0.0781426 , 0.06005379],
                         [0.05335261, 0.07264862, 0.01354238, 0.12276652, 0.13521046, 0.0631488 , 0.05184152, 0.0631488 , 0.13521046, 0.12276652, 0.01354238, 0.07264862, 0.05335261]   ])
window_wiener = np.array([14, 14, 14])

y = [np.zeros(int(point_counter)), np.zeros(int(point_counter)), np.zeros(int(point_counter))]
for i in range(0,3):
        N = window_wiener[i]//2
        y_start_pad = u[0]*np.ones(N)
        y_end_pad = u[-1]*np.ones(N)
        y[i] = np.convolve(np.append(y_start_pad, np.append(u, y_end_pad)), h_wiener[i], mode='full')
        y[i] = y[i][2*N-1:-2*N+1]  

plot_bode1 = Plot_Sig(Plot_Enum.BODE,"Bode Plot",parameters = 0)
plot_bode1.plot_sig(t,[[u,u,u],[y[0],y[1],y[2]]],[
        "$\sigma$ = 0.2", "$\sigma$ = 0.3", "$\sigma$ = 0.4"])

y1      = kalman.filter_fun(t,u,para = [2,[0,0],noise_std_dev, 5e3])[0]
y3      = kalman.filter_fun(t,u,para = [2,[0,0],noise_std_dev, 1e4])[0]
y5      = kalman.filter_fun(t,u,para = [2,[0,0],noise_std_dev, 5e4])[0]

plot_bode2 = Plot_Sig(Plot_Enum.BODE,"Bode Plot",parameters = 0)
plot_bode2.plot_sig(t,[[u,u,u],[y1,y3,y5]],[
        "process noise $\sigma$ = 1e3", 
        "process noise $\sigma$ = 1e4",
        "process noise $\sigma$ = 1e5",])

plt.show()