
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

butter = signal.butter(6,0.01)
print(butter)