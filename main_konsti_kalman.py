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

noise_std_dev   = 0.5

# 1. Filtereigenschaften auf Sinus / Polynom
point_counter = 500
sine    = Input_Function(Input_Enum.SINE, [1, 0.5, 0, 0], sampling_period = step_size, point_counter=point_counter)
polynome = Input_Function(Input_Enum.POLYNOM, [100,-150,50,0], sampling_period = step_size, point_counter=point_counter) #coefs in descending order 2x^2+1 = [2,0,1]
input_func = polynome

kalman_filter_order = 4 #2
process_std_dev = 2e4
x_start_guess =     np.array([[0], [50], [-300], [600]]) #np.array([[0], [2*np.pi*2]])

kalman  = Filter(Filter_Enum.KALMAN, parameters=None)

white   = Noise(Noise_Enum.WHITE, noise_std_dev)
brown   = Noise(Noise_Enum.BROWN, noise_std_dev)
quant   = Noise(Noise_Enum.QUANT, noise_std_dev)

cost    = Cost(Cost_Enum.MSE)

plot1  = Plot_Sig(Plot_Enum.FILTER1, "Filterung",[])

def filter_cost(para_in, t, y, para_filt, x, filter, cost):
        y_hat = filter.filter_fun(t, y, para = [para_filt[0], para_filt[1], para_filt[2], para_in])[0]
        return cost.cost(y_hat, x)
def filter_cost_diff(para_in, t, y, para_filt, x_dot, filter, cost):
        y_hat_dot = filter.filter_fun(t, y, para = [para_filt[0], para_filt[1], para_filt[2], para_in])[1]
        return cost.cost(y_hat_dot, x_dot)

t, x, x_dot = input_func.get_fun()
y_white = white.apply_noise(x)
y_brown = brown.apply_noise(x)
y_quant = quant.apply_noise(x)

kalman_para_white = minimize(filter_cost,process_std_dev,args=(t, y_white, [kalman_filter_order, x_start_guess, noise_std_dev], x, kalman, cost), method='Nelder-Mead')
x_hat_min_white = kalman.filter_fun(t,y_white,para = [kalman_filter_order,x_start_guess,noise_std_dev,abs(kalman_para_white.x)])[0]
cost_white = cost.cost(x_hat_min_white,x)
standard_cost_white = cost.cost(y_white,x)

kalman_para_brown = minimize(filter_cost,process_std_dev,args=(t, y_brown, [kalman_filter_order, x_start_guess, noise_std_dev], x, kalman, cost), method='Nelder-Mead')
x_hat_min_brown = kalman.filter_fun(t,y_brown,para = [kalman_filter_order,x_start_guess,noise_std_dev,abs(kalman_para_brown.x)])[0]
cost_brown = cost.cost(x_hat_min_brown,x)
standard_cost_brown = cost.cost(y_brown,x)

kalman_para_quant = minimize(filter_cost,process_std_dev,args=(t, y_quant, [kalman_filter_order, x_start_guess, noise_std_dev], x, kalman, cost), method='Nelder-Mead')
x_hat_min_quant = kalman.filter_fun(t,y_quant,para = [kalman_filter_order,x_start_guess,noise_std_dev,abs(kalman_para_quant.x)])[0]
cost_quant = cost.cost(x_hat_min_quant,x)
standard_cost_quant = cost.cost(y_quant,x)

box_label_white = '\n'.join((
        r'White Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'Process Noise $\sigma=%.2f$' % (abs(kalman_para_white.x), ),
        r'$MSE_{Noise}=%.5f$' % (standard_cost_white, ),
        r'$MSE_{Filter}=%.5f$' % (cost_white, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_white/standard_cost_white, )))

box_label_brown = '\n'.join((
        r'Brown Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'Process Noise $\sigma=%.2f$' % (abs(kalman_para_brown.x), ),
        r'$MSE_{Noise}=%.5f$' % (standard_cost_brown, ),
        r'$MSE_{Filter}=%.5f$' % (cost_brown, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_brown/standard_cost_brown, )))

box_label_quant = '\n'.join((
        r'Quantisation Noise',
        r'$stepsize=%.2f$' % (noise_std_dev, ),
        r'Process Noise $\sigma=%.2f$' % (abs(kalman_para_quant.x), ),
        r'$MSE_{Noise}=%.5f$' % (standard_cost_quant, ),
        r'$MSE_{Filter}=%.5f$' % (cost_quant, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_quant/standard_cost_quant, )))

plot1.plot_sig(t,[x,y_white,y_brown,y_quant,x_hat_min_white,x_hat_min_brown,x_hat_min_quant],['Input Signal',
'Signal with White Noise',
'Signal with Brown Noise',
'Signal with Quantisation Noise',
'Kalman Smoothing (White Noise)',
'Kalman Smoothing (Brown Noise)',
'Kalman Smoothing (Quantisation)',
box_label_white,box_label_brown,box_label_quant],True)


# 2. Ableitungseigenschaften auf Sinus / Polynom

y_white_dot = np.diff(y_white, append = 0)/step_size
y_brown_dot = np.diff(y_brown, append = 0)/step_size
y_quant_dot = np.diff(y_quant, append = 0)/step_size

plot2  = Plot_Sig(Plot_Enum.FILTER2, "Filterung",[])

kalman_para_white = minimize(filter_cost_diff,process_std_dev,args=(t, y_white, [kalman_filter_order, x_start_guess, noise_std_dev], x_dot, kalman, cost), method='Nelder-Mead')
x_hat_min_white = kalman.filter_fun(t,y_white,para = [kalman_filter_order,x_start_guess,noise_std_dev,abs(kalman_para_white.x)])[1]
cost_white = cost.cost(x_hat_min_white,x_dot)
standard_cost_white = cost.cost(y_white_dot,x_dot)

kalman_para_brown = minimize(filter_cost_diff,process_std_dev,args=(t, y_brown, [kalman_filter_order, x_start_guess, noise_std_dev], x_dot, kalman, cost), method='Nelder-Mead')
x_hat_min_brown = kalman.filter_fun(t,y_brown,para = [kalman_filter_order,x_start_guess,noise_std_dev,abs(kalman_para_brown.x)])[1]
cost_brown = cost.cost(x_hat_min_brown,x_dot)
standard_cost_brown = cost.cost(y_brown_dot,x_dot)

kalman_para_quant = minimize(filter_cost_diff,process_std_dev,args=(t, y_quant, [kalman_filter_order, x_start_guess, noise_std_dev], x_dot, kalman, cost), method='Nelder-Mead')
x_hat_min_quant = kalman.filter_fun(t,y_quant,para = [kalman_filter_order,x_start_guess,noise_std_dev,abs(kalman_para_quant.x)])[1]
cost_quant = cost.cost(x_hat_min_quant,x_dot)
standard_cost_quant = cost.cost(y_quant_dot,x_dot)


box_label_white = '\n'.join((
        r'White Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'Process Noise $\sigma=%.3f$' % (abs(kalman_para_white.x), ),
        r'$MSE_{Noise}=%.2f$' % (standard_cost_white, ),
        r'$MSE_{Filter}=%.2f$' % (cost_white, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_white/standard_cost_white, )))

box_label_brown = '\n'.join((
        r'Brown Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'Process Noise $\sigma=%.3f$' % (abs(kalman_para_brown.x), ),
        r'$MSE_{Noise}=%.2f$' % (standard_cost_brown, ),
        r'$MSE_{Filter}=%.2f$' % (cost_brown, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_brown/standard_cost_brown, )))

box_label_quant = '\n'.join((
        r'Quantisation Noise',
        r'$stepsize=%.2f$' % (noise_std_dev, ),
        r'Process Noise $\sigma=%.3f$' % (abs(kalman_para_quant.x), ),
        r'$MSE_{Noise}=%.2f$' % (standard_cost_quant, ),
        r'$MSE_{Filter}=%.2f$' % (cost_quant, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_quant/standard_cost_quant, )))

plot2.plot_sig(t,[x_dot,y_white_dot,y_brown_dot,y_quant_dot,x_hat_min_white,x_hat_min_brown,x_hat_min_quant],['Input Signal',
'Diff of signal with White Noise',
'Diff of signal with Brown Noise',
'Diff of signal with Quantisation Noise',
'Kalman Smoothing and Differentation',
'Kalman Smoothing and Differentation',
'Kalman Smoothing and Differentation',
box_label_white,box_label_brown,box_label_quant],True)

# Bode-Plot

# Übertragungsfunktion des Filters bestimmen


u = np.zeros(int(point_counter))
u[10:] = 1
t = np.linspace(0,1,num = int(point_counter))

y1      = kalman.filter_fun(t,u,para = [2,[0,0],noise_std_dev, 1e3])[0]
y5      = kalman.filter_fun(t,u,para = [2,[0,0],noise_std_dev, 1e4])[0]
y10     = kalman.filter_fun(t,u,para = [2,[0,0],noise_std_dev, 1e5])[0]


plot_bode = Plot_Sig(Plot_Enum.BODE,"Bode Plot",parameters = 0)

plot_bode.plot_sig(t,[[u,u,u],[y1,y5,y10]],[
        "process noise $\sigma$ = 1e3", 
        "process noise $\sigma$ = 1e4",
        "process noise $\sigma$ = 1e5",])

plt.show()