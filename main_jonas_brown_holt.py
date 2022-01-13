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

step_size       = 1e-3

noise_std_dev   = 0.1
alpha           = 0.4
beta            = 0.2
freq            = 10 / (1000 / 2) # Normalisierte Grenzfrequenz mit w = fc / (fs / 2)

bounds_fun          = ((0.0001, 0.9999),)
bounds_diff          = ((0.0001, 0.9999),(0.0001, 0.9999),)

# 1. Filtereigenschaften auf Sinus

sine    = Input_Function(Input_Enum.SINE, [1, 0.4, 0, 0], sampling_period = step_size)

exp   = Filter(Filter_Enum.BROWN_HOLT, [alpha,1])

white   = Noise(Noise_Enum.WHITE, noise_std_dev)
brown   = Noise(Noise_Enum.BROWN, noise_std_dev)
quant   = Noise(Noise_Enum.QUANT, noise_std_dev)

cost    = Cost(Cost_Enum.MSE)

plot1  = Plot_Sig(Plot_Enum.FILTER1, "Filterung",[])

def filter_cost(para_in, t, y, x, filter, cost):
        y_hat = filter.filter_fun(t, y, para = [para_in, filter.parameters[1]])
        return cost.cost(y_hat, x)
def filter_cost_diff(para_in, t, y, x_dot, filter, cost):
        y_hat_dot = filter.filter_diff(t, y, para = [para_in[0], filter.parameters[1], para_in[1]])
        return cost.cost(y_hat_dot, x_dot)

t, x, x_dot = sine.get_fun()
y_white = white.apply_noise(x)
y_brown = brown.apply_noise(x)
y_quant = quant.apply_noise(x)

alpha_min_white = minimize(filter_cost,alpha,args=(t, y_white, x, exp ,cost), bounds = bounds_fun)
x_hat_min_white = exp.filter_fun(t,y_white,para = [alpha_min_white.x,exp.parameters[1]])
cost_white = cost.cost(x_hat_min_white,x)
standard_cost_white = cost.cost(y_white,x)

alpha_min_brown = minimize(filter_cost,alpha,args=(t, y_brown, x, exp ,cost), bounds = bounds_fun)
x_hat_min_brown = exp.filter_fun(t,y_brown,para = [alpha_min_brown.x,exp.parameters[1]])
cost_brown = cost.cost(x_hat_min_brown,x)
standard_cost_brown = cost.cost(y_brown,x)

alpha_min_quant = minimize(filter_cost,alpha,args=(t, y_quant, x, exp ,cost), bounds = bounds_fun)
x_hat_min_quant = exp.filter_fun(t,y_quant,para = [alpha_min_quant.x,exp.parameters[1]])
cost_quant = cost.cost(x_hat_min_quant,x)
standard_cost_quant = cost.cost(y_quant,x)

box_label_white = '\n'.join((
        r'White Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'$\alpha=%.2f$' % (alpha_min_white.x, ),
        r'$MSE_{Noise}=%.5f$' % (standard_cost_white, ),
        r'$MSE_{Filter}=%.5f$' % (cost_white, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_white/standard_cost_white, )))

box_label_brown = '\n'.join((
        r'Brown Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'$\alpha=%.2f$' % (alpha_min_brown.x, ),
        r'$MSE_{Noise}=%.5f$' % (standard_cost_brown, ),
        r'$MSE_{Filter}=%.5f$' % (cost_brown, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_brown/standard_cost_brown, )))

box_label_quant = '\n'.join((
        r'Quantisation Noise',
        r'$stepsize=%.2f$' % (noise_std_dev, ),
        r'$\alpha=%.2f$' % (alpha_min_quant.x, ),
        r'$MSE_{Noise}=%.5f$' % (standard_cost_quant, ),
        r'$MSE_{Filter}=%.5f$' % (cost_quant, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_quant/standard_cost_quant, )))

plot1.plot_sig(t,[x,y_white,y_brown,y_quant,x_hat_min_white,x_hat_min_brown,x_hat_min_quant],['Input Signal',
'Signal with White Noise',
'Signal with Brown Noise',
'Signal with Quantisation Noise',
'Exponential Smoothing (White Noise)',
'Exponential Smoothing (Brown Noise)',
'Exponential Smoothing (Quantisation)',
box_label_white,box_label_brown,box_label_quant],True)

# 2. Ableitungseigenschaften auf Sinus

y_white_dot = np.diff(y_white, append = 0)/step_size
y_brown_dot = np.diff(y_brown, append = 0)/step_size
y_quant_dot = np.diff(y_quant, append = 0)/step_size

plot2  = Plot_Sig(Plot_Enum.FILTER2, "Filterung",[])

alpha_min_white = minimize(filter_cost_diff,(alpha, beta),args=(t, y_white, x_dot, exp ,cost), bounds = bounds_diff)
x_hat_min_white = exp.filter_diff(t,y_white,para = [alpha_min_white.x[0],exp.parameters[1],alpha_min_white.x[1]])
cost_white = cost.cost(x_hat_min_white,x_dot)
standard_cost_white = cost.cost(y_white_dot,x_dot)

alpha_min_brown = minimize(filter_cost_diff,(alpha, beta),args=(t, y_brown, x_dot, exp ,cost), bounds = bounds_diff)
x_hat_min_brown = exp.filter_diff(t,y_brown,para = [alpha_min_white.x[0],exp.parameters[1],alpha_min_white.x[1]])
cost_brown = cost.cost(x_hat_min_brown,x_dot)
standard_cost_brown = cost.cost(y_brown_dot,x_dot)

alpha_min_quant = minimize(filter_cost_diff,(alpha, beta),args=(t, y_quant, x_dot, exp ,cost), bounds = bounds_diff)
x_hat_min_quant = exp.filter_diff(t,y_quant,para = [alpha_min_white.x[0],exp.parameters[1],alpha_min_white.x[1]])
cost_quant = cost.cost(x_hat_min_quant,x_dot)
standard_cost_quant = cost.cost(y_quant_dot,x_dot)

box_label_white = '\n'.join((
        r'White Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'$\alpha=%.3f$' % (alpha_min_white.x[0], ),
        r'$\beta=%.3f$' % (alpha_min_brown.x[1], ),
        r'$MSE_{Noise}=%.2f$' % (standard_cost_white, ),
        r'$MSE_{Filter}=%.2f$' % (cost_white, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_white/standard_cost_white, )))

box_label_brown = '\n'.join((
        r'Brown Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'$\alpha=%.3f$' % (alpha_min_brown.x[0], ),
        r'$\beta=%.3f$' % (alpha_min_brown.x[1], ),
        r'$MSE_{Noise}=%.2f$' % (standard_cost_brown, ),
        r'$MSE_{Filter}=%.2f$' % (cost_brown, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_brown/standard_cost_brown, )))

box_label_quant = '\n'.join((
        r'Quantisation Noise',
        r'$stepsize=%.2f$' % (noise_std_dev, ),
        r'$\alpha=%.3f$' % (alpha_min_quant.x[0], ),
        r'$\beta=%.3f$' % (alpha_min_brown.x[1], ),
        r'$MSE_{Noise}=%.2f$' % (standard_cost_quant, ),
        r'$MSE_{Filter}=%.2f$' % (cost_quant, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_quant/standard_cost_quant, )))

plot2.plot_sig(t,[x_dot,y_white_dot,y_brown_dot,y_quant_dot,x_hat_min_white,x_hat_min_brown,x_hat_min_quant],['Input Signal',
'Diff of signal with White Noise',
'Diff of signal with Brown Noise',
'Diff of signal with Quantisation Noise',
'Exponential Smoothing and Differentation',
'Exponential Smoothing and Differentation',
'Exponential Smoothing and Differentation',
box_label_white,box_label_brown,box_label_quant],True)

plt.show()