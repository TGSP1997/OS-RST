
from re import S
from tkinter import Y
from matplotlib.pyplot import show
from matplotlib import ticker, cm
from numpy.lib.polynomial import poly
import scipy
from scipy.optimize import minimize
from scipy.fft import fft, ifft, fftfreq
from scipy import signal
from numpy import ma
import matplotlib.pyplot as plt


from input_function import *
from noises import *
from filter import *
from plot_sig import *
from cost import *

step_size       = 1e-2
point_counter   = int(1/step_size)


noise_std_dev   = 0.1
alpha           = 0.4
beta            = 0.2
freq            = 10 / (1000 / 2) # Normalisierte Grenzfrequenz mit w = fc / (fs / 2)

bounds_fun      = ((0.01, 0.99),)
bounds_diff     = ((0.01, 0.99),(0.01, 0.99),)

order = 1

L = 50

u = np.ones(int(point_counter))
for i in range(0, 11):
    u[i] = 0

t = range(0,int(point_counter))

y = np.zeros(int(point_counter))

for i in range(1, point_counter):
    if i<L:
        y[i] = np.sum(u[0:i])/(L-1)
    else:
        y[i] = y[i-1] + (1/(L-1)) * (u[i]-u[i-L])


exp   = Filter(Filter_Enum.BROWN_HOLT, [0.8,1])

y_exp = exp.filter_fun(t,u)


fig = plt.figure(figsize=(10, 5), dpi=120, constrained_layout=True)

plt.plot(t, u,'r--', linewidth = 3, label="Step")
plt.plot(t, y,'b', linewidth = 1, label="Moving average $L=50$")
plt.plot(t, y_exp,'k', linewidth = 2, label=r"Exponential Smoothing $\alpha=0.8$")
plt.legend(loc="lower right")
plt.ylabel('value', fontsize=16)
plt.xlim(min(t),max(t))
plt.tick_params(
    axis="x",
    which="both",
    bottom=False,
    top = False,
    labelbottom = False
)
plt.grid(True)
plt.yticks(fontsize=14)
plt.xlabel('steps', fontsize=16)
plt.tick_params(
        axis="x",
        which="both",
        bottom=True,
        top = False,
        labelbottom = True
    )
plt.xticks(fontsize=14)
fig.suptitle("Step Response Exponential Smoothing", fontsize=16)

plt.show()

exp   = Filter(Filter_Enum.BROWN_HOLT, [0.5,1])
exp2   = Filter(Filter_Enum.BROWN_HOLT, [0.5,2])

white = Noise(Noise_Enum.WHITE, 0.2)

sine    = Input_Function(Input_Enum.SINE, [1, 1, 0, 0], sampling_period = step_size, point_counter = point_counter)
t, x, x_dot = sine.get_fun()

y = white.apply_noise(x)

x_hat1 = exp.filter_fun(t,y)
x_hat2 = exp2.filter_fun(t,y)



fig = plt.figure(figsize=(10, 5), dpi=120, constrained_layout=True)
plt.plot(t, x,'r--', linewidth = 2, label="Step")
plt.plot(t, y,'b:', linewidth = 1, label="Noisy sin")
plt.plot(t, x_hat1,'b--', linewidth = 1, label=r"Exponential Smoothing $\alpha=0.5$")
plt.plot(t, x_hat2,'k', linewidth = 2, label=r"Double Exp. Smoothing $\alpha=0.5$")
plt.legend(loc="upper right")
plt.ylabel('value', fontsize=16)
plt.xlim(min(t),max(t))
plt.tick_params(
    axis="x",
    which="both",
    bottom=False,
    top = False,
    labelbottom = False
)
plt.grid(True)
plt.yticks(fontsize=14)
plt.xlabel('steps', fontsize=16)
plt.tick_params(
        axis="x",
        which="both",
        bottom=True,
        top = False,
        labelbottom = True
    )
plt.xticks(fontsize=14)
fig.suptitle("Step Response Exponential Smoothing", fontsize=16)
plt.show()

butter = signal.butter(4,0.01)
print(butter)
butter = signal.butter(5,0.01)
print(butter)
butter = signal.butter(6,0.01)
print(butter)
butter = signal.butter(7,0.01)
print(butter)

step_size       = 2e-3
point_counter   = int(1/step_size)

exp = Filter(Filter_Enum.BROWN_HOLT,[0.091,1,0.034])

noise_std_dev = 0.5
white   = Noise(Noise_Enum.WHITE, noise_std_dev)
brown   = Noise(Noise_Enum.BROWN, noise_std_dev)
quant   = Noise(Noise_Enum.QUANT, noise_std_dev)

sine    = Input_Function(Input_Enum.SINE, [1, 0.5, 0, 0], sampling_period = step_size, point_counter = point_counter)

plot2  = Plot_Sig(Plot_Enum.FILTER2, "Exponential Smoothing | Derivative Harmonic Signal",[])

cost    = Cost(Cost_Enum.MSE)

t, x, x_dot = sine.get_fun()
y_white = white.apply_noise(x)
y_brown = brown.apply_noise(x)
y_quant = quant.apply_noise(x)

x_hat_min_white = exp.filter_diff(t,y_white)

cost_white = cost.cost(x_hat_min_white,x)
standard_cost_white = cost.cost(y_white,x)

y_white_dot = np.diff(y_white, append = 0)/step_size
y_brown_dot = np.diff(y_brown, append = 0)/step_size
y_quant_dot = np.diff(y_quant, append = 0)/step_size

alpha_min_white = [0.091,0.057]

x_hat_min_white = exp.filter_diff(t,y_white,para = [alpha_min_white[0],exp.parameters[1],alpha_min_white[1]])  # Startwert x_dot[0] wird übergeben um die Konvergierungszeit zu verkürzen
cost_white = cost.cost(x_hat_min_white,x_dot)
standard_cost_white = cost.cost(y_white_dot,x_dot)


alpha_min_brown = [0.990,0.034]

x_hat_min_brown = exp.filter_diff(t,y_brown,para = [alpha_min_brown[0],exp.parameters[1],alpha_min_brown[1]])  # Startwert x_dot[0] wird übergeben um die Konvergierungszeit zu verkürzen
cost_brown = cost.cost(x_hat_min_brown,x_dot)
standard_cost_brown = cost.cost(y_brown_dot,x_dot)

alpha_min_quant = [0.036,0.166]

x_hat_min_quant = exp.filter_diff(t,y_quant,para = [alpha_min_quant[0],exp.parameters[1],alpha_min_quant[1]])  # Startwert x_dot[0] wird übergeben um die Konvergierungszeit zu verkürzen
cost_quant = cost.cost(x_hat_min_quant,x_dot)
standard_cost_quant = cost.cost(y_quant_dot,x_dot)

box_label_white = '\n'.join((
                r'White Noise',
                r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
                r'$\alpha=%.3f$' % (alpha_min_white[0], ),
                r'$\beta=%.3f$' % (alpha_min_white[1], ),
                r'$MSE_{Filter}=%.2e$' % (cost_white, ),
                r'$MSE_{Noise}=%.2e$' % (standard_cost_white, ),
                r'$r_{MSE}=%.2f$ %%' % (100*cost_white/standard_cost_white, )))

box_label_brown = '\n'.join((
        r'Brown Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'$\alpha=%.3f$' % (alpha_min_brown[0], ),
        r'$\beta=%.3f$' % (alpha_min_brown[1], ),
        r'$MSE_{Filter}=%.2e$' % (cost_brown, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_brown, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_brown/standard_cost_brown, )))

box_label_quant = '\n'.join((
        r'Quantisation Noise',
        r'$stepsize=%.2f$' % (noise_std_dev, ),
        r'$\alpha=%.3f$' % (alpha_min_quant[0], ),
        r'$\beta=%.3f$' % (alpha_min_brown[1], ),
        r'$MSE_{Filter}=%.2e$' % (cost_quant, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_quant, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_quant/standard_cost_quant, )))

plot2.plot_sig(t,[x_dot,y_white_dot,y_brown_dot,y_quant_dot,x_hat_min_white,x_hat_min_brown,x_hat_min_quant],[
r'$\frac{df}{dt}(t) = \left(\frac{2\pi}{0.5 \mathrm{s}}\right)\mathrm{cos}\left(2\pi\cdot\frac{t}{0.5 \mathrm{s}}\right)$',
'Difference of noisy signal',
'Difference of noisy signal',
'Difference of noisy signal',
'Filtered & derived signal | '+ str(order) + '. order exp. smoothing',
'Filtered & derived signal | '+ str(order) + '. order exp. smoothing',
'Filtered & derived signal | '+ str(order) + '. order exp. smoothing',
box_label_white,box_label_brown,box_label_quant],True)
plt.show()