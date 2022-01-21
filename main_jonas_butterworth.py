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


step_size       = 2e-3
point_counter   = int(1/step_size)

noise_std_dev   = 0.1

order           = 4
freq            = 10 / (1000 / 2) # Normalisierte Grenzfrequenz mit w = fc / (fs / 2)

bounds          = ((0.001, 0.9),) # Komische Tupeldarstellung damit Minimize Glücklich ist.


sine    = Input_Function(Input_Enum.SINE, [1, 0.5, 0, 0], sampling_period = step_size, point_counter = point_counter)

poly    = Input_Function(Input_Enum.POLYNOM, [4,-6,3,0], sampling_period = step_size, point_counter = point_counter)

# 1. Filtereigenschaften auf Sinus

butter   = Filter(Filter_Enum.BUTTERWORTH, [order,freq])

white   = Noise(Noise_Enum.WHITE, noise_std_dev)
brown   = Noise(Noise_Enum.BROWN, noise_std_dev)
quant   = Noise(Noise_Enum.QUANT, noise_std_dev)

cost    = Cost(Cost_Enum.MSE)

plot1  = Plot_Sig(Plot_Enum.FILTER1, "Butterworth | Harmonic Signal",[])

plot2  = Plot_Sig(Plot_Enum.FILTER2, "Butterworth | Derivative Harmonic Signal",[])

def filter_cost(para_in, t, y, x, filter, cost):
        y_hat = filter.filter_fun(t, y, para = [filter.parameters[0], para_in])
        return cost.cost(y_hat, x) 
def filter_cost_diff(para_in, t, y, x_dot, filter, cost):
        y_hat_dot = filter.filter_diff(t, y, para = [filter.parameters[0], para_in])
        return cost.cost(y_hat_dot, x_dot)

t, x, x_dot = sine.get_fun()
y_white = white.apply_noise(x)
y_brown = brown.apply_noise(x)
y_quant = quant.apply_noise(x)

freq_min_white = minimize(filter_cost,freq,args=(t, y_white, x, butter ,cost), bounds= bounds)
x_hat_min_white = butter.filter_fun(t,y_white,para = [butter.parameters[0],freq_min_white.x])
cost_white = cost.cost(x_hat_min_white,x)
standard_cost_white = cost.cost(y_white,x)

freq_min_brown = minimize(filter_cost,freq,args=(t, y_brown, x, butter ,cost), bounds= bounds)
x_hat_min_brown = butter.filter_fun(t,y_brown,para = [butter.parameters[0],freq_min_brown.x])
cost_brown = cost.cost(x_hat_min_brown,x)
standard_cost_brown = cost.cost(y_brown,x)

freq_min_quant = minimize(filter_cost,freq,args=(t, y_quant, x, butter ,cost), bounds= bounds)
x_hat_min_quant = butter.filter_fun(t,y_quant,para = [butter.parameters[0],freq_min_quant.x])
cost_quant = cost.cost(x_hat_min_quant,x)
standard_cost_quant = cost.cost(y_quant,x)

box_label_white = '\n'.join((
        r'White Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'$f_{-3\,\mathrm{dB}}=%.2f \frac{f}{f_S}$' % (freq_min_white.x, ),
        r'$MSE_{Filter}=%.2e$' % (cost_white, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_white, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_white/standard_cost_white, )))

box_label_brown = '\n'.join((
        r'Brown Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'$f_{-3\,\mathrm{dB}}=%.2f \frac{f}{f_S}$' % (freq_min_brown.x, ),
        r'$MSE_{Filter}=%.2e$' % (cost_brown, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_brown, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_brown/standard_cost_brown, )))

box_label_quant = '\n'.join((
        r'Quantisation Noise',
        r'$stepsize=%.2f$' % (noise_std_dev, ),
        r'$f_{-3\,\mathrm{dB}}=%.2f \frac{f}{f_S}$' % (freq_min_quant.x, ),
        r'$MSE_{Filter}=%.2e$' % (cost_quant, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_quant, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_quant/standard_cost_quant, )))

plot1.plot_sig(t,[x,y_white,y_brown,y_quant,x_hat_min_white,x_hat_min_brown,x_hat_min_quant],[
r'$f(t) = \mathrm{sin}\left(2\pi\cdot\frac{t}{0.5\,\mathrm{s}}\right)$',
'Noisy signal',
'Noisy signal',
'Noisy signal',
'Filtered signal | '+ str(order) + '. order BW-Filter',
'Filtered signal | '+ str(order) + '. order BW-Filter',
'Filtered signal | '+ str(order) + '. order BW-Filter',
box_label_white,box_label_brown,box_label_quant],True)


# 2. Ableitungseigenschaften auf Sinus

y_white_dot = np.diff(y_white, append = 0)/step_size
y_brown_dot = np.diff(y_brown, append = 0)/step_size
y_quant_dot = np.diff(y_quant, append = 0)/step_size

freq_min_white = minimize(filter_cost_diff,freq,args=(t, y_white, x_dot, butter ,cost), bounds = bounds)
x_hat_min_white = butter.filter_diff(t,y_white,para = [butter.parameters[0],freq_min_white.x]) 
cost_white = cost.cost(x_hat_min_white,x_dot)
standard_cost_white = cost.cost(y_white_dot,x_dot)

freq_min_brown = minimize(filter_cost_diff,freq,args=(t, y_brown, x_dot, butter ,cost), bounds = bounds)
x_hat_min_brown = butter.filter_diff(t,y_brown,para = [butter.parameters[0],freq_min_white.x]) 
cost_brown = cost.cost(x_hat_min_brown,x_dot)
standard_cost_brown = cost.cost(y_brown_dot,x_dot)

freq_min_quant = minimize(filter_cost_diff,freq,args=(t, y_quant, x_dot, butter ,cost), bounds = bounds)
x_hat_min_quant = butter.filter_diff(t,y_quant,para = [butter.parameters[0],freq_min_white.x])  
cost_quant = cost.cost(x_hat_min_quant,x_dot)
standard_cost_quant = cost.cost(y_quant_dot,x_dot)

box_label_white = '\n'.join((
        r'White Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'$f_{-3\,\mathrm{dB}}=%.2f \frac{f}{f_S}$' % (freq_min_white.x, ),
        r'$MSE_{Filter}=%.2e$' % (cost_white, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_white, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_white/standard_cost_white, )))

box_label_brown = '\n'.join((
        r'Brown Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'$f_{-3\,\mathrm{dB}}=%.2f \frac{f}{f_S}$' % (freq_min_brown.x, ),
        r'$MSE_{Filter}=%.2e$' % (cost_brown, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_brown, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_brown/standard_cost_brown, )))

box_label_quant = '\n'.join((
        r'Quantisation Noise',
        r'$stepsize=%.2f$' % (noise_std_dev, ),
        r'$f_{-3\,\mathrm{dB}}=%.2f \frac{f}{f_S}$' % (freq_min_quant.x, ),
        r'$MSE_{Filter}=%.2e$' % (cost_quant, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_quant, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_quant/standard_cost_quant, )))

plot2.plot_sig(t,[x_dot,y_white_dot,y_brown_dot,y_quant_dot,x_hat_min_white,x_hat_min_brown,x_hat_min_quant],[
r'$\frac{df}{dt}(t) = \left(\frac{2\pi}{0.5 \mathrm{s}}\right)\mathrm{cos}\left(2\pi\cdot\frac{t}{0.5 \mathrm{s}}\right)$',
'Difference of noisy signal',
'Difference of noisy signal',
'Difference of noisy signal',
'Filtered & derived signal | '+ str(order) + '. order BW-Filter',
'Filtered & derived signal | '+ str(order) + '. order BW-Filter',
'Filtered & derived signal | '+ str(order) + '. order BW-Filter',
box_label_white,box_label_brown,box_label_quant],True)

# 3. Filtereigenschaften auf Polynom

plot1  = Plot_Sig(Plot_Enum.FILTER3, "Butterworth | Polynomial Signal",[])

plot2  = Plot_Sig(Plot_Enum.FILTER4, "Butterworth | Derivative Polynomial Signal",[])

white   = Noise(Noise_Enum.WHITE, noise_std_dev)
brown   = Noise(Noise_Enum.BROWN, noise_std_dev)
quant   = Noise(Noise_Enum.QUANT, noise_std_dev)

t, x, x_dot = poly.get_fun()
y_white = white.apply_noise(x)
y_brown = brown.apply_noise(x)
y_quant = quant.apply_noise(x)

freq_min_white = minimize(filter_cost,freq,args=(t, y_white, x, butter, cost), bounds = bounds)
x_hat_min_white = butter.filter_fun(t,y_white,para = [butter.parameters[0], freq_min_white.x])
cost_white = cost.cost(x_hat_min_white,x)
standard_cost_white = cost.cost(y_white,x)

freq_min_brown = minimize(filter_cost,freq,args=(t, y_brown, x, butter, cost), bounds = bounds)
x_hat_min_brown = butter.filter_fun(t,y_brown,para = [butter.parameters[0], freq_min_brown.x])
cost_brown = cost.cost(x_hat_min_brown,x)
standard_cost_brown = cost.cost(y_brown,x)

freq_min_quant = minimize(filter_cost,freq,args=(t, y_quant, x, butter, cost), bounds = bounds)
x_hat_min_quant = butter.filter_fun(t,y_quant,para = [butter.parameters[0], freq_min_quant.x])
cost_quant = cost.cost(x_hat_min_quant,x)
standard_cost_quant = cost.cost(y_quant,x)

box_label_white = '\n'.join((
        r'White Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'$f_{-3\,\mathrm{dB}}=%.2f \frac{f}{f_S}$' % (freq_min_white.x, ),
        r'$MSE_{Filter}=%.2e$' % (cost_white, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_white, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_white/standard_cost_white, )))

box_label_brown = '\n'.join((
        r'Brown Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'$f_{-3\,\mathrm{dB}}=%.2f \frac{f}{f_S}$' % (freq_min_brown.x, ),
        r'$MSE_{Filter}=%.2e$' % (cost_brown, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_brown, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_brown/standard_cost_brown, )))

box_label_quant = '\n'.join((
        r'Quantisation Noise',
        r'$stepsize=%.2f$' % (noise_std_dev, ),
        r'$f_{-3\,\mathrm{dB}}=%.2f \frac{f}{f_S}$' % (freq_min_quant.x, ),
        r'$MSE_{Filter}=%.2e$' % (cost_quant, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_quant, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_quant/standard_cost_quant, )))

plot1.plot_sig(t,[x,y_white,y_brown,y_quant,x_hat_min_white,x_hat_min_brown,x_hat_min_quant],[
r'$f(t) = \frac{4}{\mathrm{s}^3}\cdot t^3 - \frac{6}{\mathrm{s}^2}\cdot t^2 + \frac{3}{\mathrm{s}}\cdot t + 0$',
'Noisy signal',
'Noisy signal',
'Noisy signal',
'Filtered signal | '+ str(order) + '. order BW-Filter',
'Filtered signal | '+ str(order) + '. order BW-Filter',
'Filtered signal | '+ str(order) + '. order BW-Filter',
box_label_white,box_label_brown,box_label_quant],True)

# 4. Ableitungseigenschaften auf Polynom

y_white_dot = np.diff(y_white, append = 0)/step_size
y_brown_dot = np.diff(y_brown, append = 0)/step_size
y_quant_dot = np.diff(y_quant, append = 0)/step_size

freq_min_white = minimize(filter_cost_diff, freq, args=(t, y_white, x_dot, butter ,cost), bounds = bounds)
x_hat_min_white = butter.filter_diff(t,y_white,para = [butter.parameters[0], freq_min_white.x])
cost_white = cost.cost(x_hat_min_white,x_dot)
standard_cost_white = cost.cost(y_white_dot,x_dot)

freq_min_brown = minimize(filter_cost_diff, freq, args=(t, y_brown, x_dot, butter ,cost), bounds = bounds)
x_hat_min_brown = butter.filter_diff(t,y_brown,para = [butter.parameters[0], freq_min_white.x])
cost_brown = cost.cost(x_hat_min_brown,x_dot)
standard_cost_brown = cost.cost(y_brown_dot,x_dot)

freq_min_quant = minimize(filter_cost_diff, freq, args=(t, y_quant, x_dot, butter ,cost), bounds = bounds)
x_hat_min_quant = butter.filter_diff(t,y_quant,para = [butter.parameters[0], freq_min_white.x])
cost_quant = cost.cost(x_hat_min_quant,x_dot)
standard_cost_quant = cost.cost(y_quant_dot,x_dot)

box_label_white = '\n'.join((
        r'White Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'$f_{-3\,\mathrm{dB}}=%.2f \frac{f}{f_S}$' % (freq_min_white.x, ),
        r'$MSE_{Filter}=%.2e$' % (cost_white, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_white, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_white/standard_cost_white, )))

box_label_brown = '\n'.join((
        r'Brown Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'$f_{-3\,\mathrm{dB}}=%.2f \frac{f}{f_S}$' % (freq_min_brown.x, ),
        r'$MSE_{Filter}=%.2e$' % (cost_brown, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_brown, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_brown/standard_cost_brown, )))

box_label_quant = '\n'.join((
        r'Quantisation Noise',
        r'$stepsize=%.2f$' % (noise_std_dev, ),
        r'$f_{-3\,\mathrm{dB}}=%.2f \frac{f}{f_S}$' % (freq_min_quant.x, ),
        r'$MSE_{Filter}=%.2e$' % (cost_quant, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_quant, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_quant/standard_cost_quant, )))

plot2.plot_sig(t,[x_dot,y_white_dot,y_brown_dot,y_quant_dot,x_hat_min_white,x_hat_min_brown,x_hat_min_quant],[
r'$f(t) = \frac{12}{\mathrm{s}^3}\cdot t^2 - \frac{12}{\mathrm{s}^2}\cdot t + \frac{3}{\mathrm{s}}$',
'Difference of noisy signal',
'Difference of noisy signal',
'Difference of noisy signal',
'Filtered & derived signal | '+ str(order) + '. order BW-Filter',
'Filtered & derived signal | '+ str(order) + '. order BW-Filter',
'Filtered & derived signal | '+ str(order) + '. order BW-Filter',
box_label_white,box_label_brown,box_label_quant],True)

# 10. Bode-Plot

# Übertragungsfunktion des Filters bestimmen

butter1   = Filter(Filter_Enum.BUTTERWORTH, [1,freq])
butter2   = Filter(Filter_Enum.BUTTERWORTH, [2,freq])
butter3   = Filter(Filter_Enum.BUTTERWORTH, [3,freq])
butter4   = Filter(Filter_Enum.BUTTERWORTH, [4,freq])
butter5   = Filter(Filter_Enum.BUTTERWORTH, [5,freq])

u = np.zeros(int(point_counter))
u[10] = 1
t = np.linspace(0,1,num = int(point_counter))

o1      = butter1.filter_diff(t,u)
o2      = butter2.filter_diff(t,u)
o3      = butter3.filter_diff(t,u)
o4      = butter4.filter_diff(t,u)
o5      = butter5.filter_diff(t,u)


plot_bode = Plot_Sig(Plot_Enum.BODE,"Bode Plot",parameters = 0)

plot_bode.plot_sig(t,[[u,u,u,u,u],[o1,o2,o3,o4,o5]],[
        "1st order", 
        "2nd order",
        "3rd order",
        "4th order",
        "5th order",])

plt.show()


#sos = signal.butter(order, freq, output='sos')
#derivative = signal.zpk2sos([1],[-1],2/t[1])  # Second order section derivative | s= (z-1)/(z+1)
#sos= np.append(sos,derivative,axis=0)


#zi = signal.sosfilt_zi(sos)                 # Initial conditions

#y_white = np.linspace(0,1,point_counter)
#y_white = np.multiply(np.ones(point_counter),1)

#print(zi)

#y_dot_hat, z_f = signal.sosfilt(sos,y_white, zi=zi)

#print(z_f)

#y_dot_hat, z_f = signal.sosfilt(sos,y_white, zi=z_f)

#y_dot_hat = np.multiply(np.diff(y_hat, prepend=0), 1/t[1])



#plt.plot(t,y_dot_hat)
#plt.plot(t,y_hat)
#plt.plot(t,y_white)
#plt.show()
