from plot_sig import *
from filter import *

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

point_counter=10000

u = np.zeros(int(point_counter))
u[10] = 1

############# Wiener ####################

# wiener
h_wiener = np.array([    [0.06005379, 0.0781426 , 0.01643154, 0.12433697, 0.13439967, 0.05805149, 0.04271674, 0.05805149, 0.13439967, 0.12433697, 0.01643154, 0.0781426 , 0.06005379],
                         [0.05335261, 0.07264862, 0.01354238, 0.12276652, 0.13521046, 0.0631488 , 0.05184152, 0.0631488 , 0.13521046, 0.12276652, 0.01354238, 0.07264862, 0.05335261],   
                         [0.04968118, 0.06947908, 0.0115695 , 0.1213343 , 0.13475785, 0.06456928, 0.0550195 , 0.06456928, 0.13475785, 0.1213343 , 0.0115695 , 0.06947908, 0.04968118]])
window_wiener = np.array([14, 14, 14])

y = [np.zeros(int(point_counter)), np.zeros(int(point_counter)), np.zeros(int(point_counter))]
for i in range(0,3):
        N = window_wiener[i]//2
        y_start_pad = u[0]*np.ones(N)
        y_end_pad = u[-1]*np.ones(N)
        y[i] = np.convolve(np.append(y_start_pad, np.append(u, y_end_pad)), h_wiener[i], mode='full')
        y[i] = y[i][2*N-1:-2*N+1] 

y_wiener = y[1]

#################### Kalman ##############

kalman  = Filter(Filter_Enum.KALMAN, parameters=None)

t = np.linspace(0,1,num = int(point_counter))

y_kalman      = kalman.filter_fun(t,u,para = [2,[0,0],0.1, 1e4])[0]



############ Savitzy Golay #########################


t = np.linspace(0,1,num = int(point_counter))
from scipy.signal import savgol_filter

savgol  = Filter(Filter_Enum.SAVGOL, parameters=None) #para=[m,polorder,diff=number]


#y_savgol      = savgol.filter_fun(t,u,para = [ 10 , 5 ])

y_savgol = savgol_filter(u, 7, 3 )

############### Expotential Smoothing ###################

alpha = 0.2
beta = 0.2

exp =    Filter(Filter_Enum.BROWN_HOLT, [alpha,1,beta])

t = np.linspace(0,1,num = int(point_counter))

y_exp      = exp.filter_fun(t,u)

plot_bode = Plot_Sig(Plot_Enum.BODE,"Bode Plot Exp. Smoothing and Derivative",parameters = 0)

############## Butterworth ########################

freq = 10 / (1000 / 2)

butter   = Filter(Filter_Enum.BUTTERWORTH, [6,freq])

t = np.linspace(0,1,num = int(point_counter))

y_butter      = butter.filter_fun(t,u)

# Darstellung

print(y_kalman)

plot_bode = Plot_Sig(Plot_Enum.BODE,"Bode Plot Filter Comparison",parameters = 0)
plot_bode.plot_sig(t,[[u,u,u,u,u],[y_wiener,y_kalman, y_savgol,y_exp,y_butter]],[
        r"Wiener,$\sigma = 0.4$",
        r"Kalman, Prozessrauschen $\sigma = 1e4$",
        r"Savitzy-Golay, Fensterbr. = 7, poly = 3",
        r"Exponentiale Gl., $\alpha = \beta = 0.2$",
        r"Butterworth, $\frac{f}{f_s} = 0.01$",
        ],ylim = [-40,10])

plt.show()