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

step_size       = 2.0e-3
point_counter = 500
noise_std_dev   = 0.1

# 0. Minimize function for window length of Wiener time implementation
def minimize_wiener_window(t,y,x,filter,diff,cost):
        window_length = 2
        cost_best = float("inf")
        cost_array = []
        for i in range(window_length, point_counter, 2):
                if diff == 0:
                        x_hat = filter.filter_fun(t,y,[noise_std_dev, i])[0]
                else:
                        x_hat = filter.filter_diff(t,y,[noise_std_dev, i])[0]
                cost_now = cost.cost(x_hat,x)
                cost_array.append(cost_now)
                if cost_now < cost_best:
                        cost_best = cost_now 
                        window_length = i
        return window_length

# 1. Filtereigenschaften auf Sinus / Polynom
sine    = Input_Function(Input_Enum.SINE, [1, 0.5, 0, 0], sampling_period = step_size, point_counter=point_counter)
polynome = Input_Function(Input_Enum.POLYNOM, [100,-150,50,0], sampling_period = step_size, point_counter=point_counter) #coefs in descending order 2x^2+1 = [2,0,1]
input_func = sine

wiener   = Filter(Filter_Enum.WIENER, [])

white   = Noise(Noise_Enum.WHITE, noise_std_dev)
brown   = Noise(Noise_Enum.BROWN, noise_std_dev)
quant   = Noise(Noise_Enum.QUANT, noise_std_dev)

cost    = Cost(Cost_Enum.MSE)

plot1  = Plot_Sig(Plot_Enum.FILTER1, "Filterung",[])

t, x, x_dot = input_func.get_fun()
y_white = white.apply_noise(x)
y_brown = brown.apply_noise(x)
y_quant = quant.apply_noise(x)


window_len_white = minimize_wiener_window(t,y_white,x,wiener,0,cost)
x_hat_min_white = wiener.filter_fun(t,y_white,[noise_std_dev, window_len_white])[0]
cost_white = cost.cost(x_hat_min_white,x)
standard_cost_white = cost.cost(y_white,x)

window_len_brown = minimize_wiener_window(t,y_brown,x,wiener,0,cost)
x_hat_min_brown = wiener.filter_fun(t,y_brown,[noise_std_dev, window_len_brown])[0]
cost_brown = cost.cost(x_hat_min_brown,x)
standard_cost_brown = cost.cost(y_brown,x)

window_len_quant = minimize_wiener_window(t,y_quant,x,wiener,0,cost)
x_hat_min_quant = wiener.filter_fun(t,y_quant,[noise_std_dev, window_len_quant])[0]
cost_quant = cost.cost(x_hat_min_quant,x)
standard_cost_quant = cost.cost(y_quant,x)

box_label_white = '\n'.join((
        r'White Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'window_length$=%d$' % (window_len_white, ),
        r'$MSE_{Noise}=%.5f$' % (standard_cost_white, ),
        r'$MSE_{Filter}=%.5f$' % (cost_white, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_white/standard_cost_white, )))

box_label_brown = '\n'.join((
        r'Brown Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'window_length$=%d$' % (window_len_brown, ),
        r'$MSE_{Noise}=%.5f$' % (standard_cost_brown, ),
        r'$MSE_{Filter}=%.5f$' % (cost_brown, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_brown/standard_cost_brown, )))

box_label_quant = '\n'.join((
        r'Quantisation Noise',
        r'$stepsize=%.2f$' % (noise_std_dev, ),
        r'window_length$=%d$' % (window_len_quant, ),
        r'$MSE_{Noise}=%.5f$' % (standard_cost_quant, ),
        r'$MSE_{Filter}=%.5f$' % (cost_quant, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_quant/standard_cost_quant, )))

plot1.plot_sig(t,[x,y_white,y_brown,y_quant,x_hat_min_white,x_hat_min_brown,x_hat_min_quant],['Input Signal',
'Signal with White Noise',
'Signal with Brown Noise',
'Signal with Quantisation Noise',
'Wiener Smoothing (White Noise)',
'Wiener Smoothing (Brown Noise)',
'Wiener Smoothing (Quantisation)',
box_label_white,box_label_brown,box_label_quant],True)


# 2. Ableitungseigenschaften auf Sinus / Polynom

y_white_dot = np.diff(y_white, append = 0)/step_size
y_brown_dot = np.diff(y_brown, append = 0)/step_size
y_quant_dot = np.diff(y_quant, append = 0)/step_size

plot2  = Plot_Sig(Plot_Enum.FILTER2, "Filterung",[])

x_hat_dot_white = wiener.filter_diff(t,y_white,[noise_std_dev, window_len_white])[0]
cost_white = cost.cost(x_hat_dot_white,x_dot)
standard_cost_white = cost.cost(y_white_dot,x_dot)

x_hat_dot_brown = wiener.filter_diff(t,y_brown,[noise_std_dev, window_len_brown])[0]
cost_brown = cost.cost(x_hat_dot_brown,x_dot)
standard_cost_brown = cost.cost(y_brown_dot,x_dot)

x_hat_dot_quant = wiener.filter_diff(t,y_quant,[noise_std_dev, window_len_quant])[0]
cost_quant = cost.cost(x_hat_dot_quant,x_dot)
standard_cost_quant = cost.cost(y_quant_dot,x_dot)

box_label_white = '\n'.join((
        r'White Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'window_length$=%d$' % (window_len_white, ),
        r'$MSE_{Noise}=%.2f$' % (standard_cost_white, ),
        r'$MSE_{Filter}=%.2f$' % (cost_white, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_white/standard_cost_white, )))

box_label_brown = '\n'.join((
        r'Brown Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'window_length$=%d$' % (window_len_brown, ),
        r'$MSE_{Noise}=%.2f$' % (standard_cost_brown, ),
        r'$MSE_{Filter}=%.2f$' % (cost_brown, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_brown/standard_cost_brown, )))

box_label_quant = '\n'.join((
        r'Quantisation Noise',
        r'$stepsize=%.2f$' % (noise_std_dev, ),
        r'window_length$=%d$' % (window_len_quant, ),
        r'$MSE_{Noise}=%.2f$' % (standard_cost_quant, ),
        r'$MSE_{Filter}=%.2f$' % (cost_quant, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_quant/standard_cost_quant, )))

plot2.plot_sig(t,[x_dot,y_white_dot,y_brown_dot,y_quant_dot,x_hat_dot_white,x_hat_dot_brown,x_hat_dot_quant],['Input Signal',
'Diff of signal with White Noise',
'Diff of signal with Brown Noise',
'Diff of signal with Quantisation Noise',
'Wiener Smoothing and Differentation',
'Wiener Smoothing and Differentation',
'Wiener Smoothing and Differentation',
box_label_white,box_label_brown,box_label_quant],True)

# Bode-Plot

# Übertragungsfunktion des Filters bestimmen

'''
u = np.zeros(int(point_counter))
u[10:] = 1
t = np.linspace(0,1,num = int(point_counter))

y1      = wiener.filter_fun(t,u, [0.1, window_len_white])[0]
y5      = wiener.filter_fun(t,u, [0.5, window_len_white])[0]
y10     = wiener.filter_fun(t,u, [1.0, window_len_white])[0]


plot_bode = Plot_Sig(Plot_Enum.BODE,"Bode Plot",parameters = 0)

plot_bode.plot_sig(t,[[u,u,u],[y1,y5,y10]],[
        "$\sigma$ = 0.1", 
        "$\sigma$ = 0.5",
        "$\sigma$ = 1.0",])
'''

plt.show()