from scipy.optimize import minimize
from scipy.fft import fft, ifft, fftfreq
from scipy import signal

from input_function import *
from noises import *
from filter import *
from plot_sig import *
from cost import *

noise_std_dev   = 0.1
alpha           = 0.4
freq            = 10 / (1000 / 2) # Normalisierte Grenzfrequenz mit w = fc / (fs / 2)

sine    = Input_Function(Input_Enum.SINE, [1, 0.2, 0, 0])
brown   = Filter(Filter_Enum.BROWN_HOLT, alpha)
butter2  = Filter(Filter_Enum.BUTTERWORTH, [2, 2*np.pi*freq])
butter3  = Filter(Filter_Enum.BUTTERWORTH, [3, 2*np.pi*freq])
butter4  = Filter(Filter_Enum.BUTTERWORTH, [4, 2*np.pi*freq])
butter5  = Filter(Filter_Enum.BUTTERWORTH, [5, 2*np.pi*freq])
cost    = Cost(Cost_Enum.MSE)
plot    = Plot_Sig(Plot_Enum.SUBPLOT,"Butter",0)
plot_s  = Plot_Sig(Plot_Enum.SLIDER, "Detailed View with Slider",[])#only one slider window can be open at a time
white   = Noise(Noise_Enum.WHITE, noise_std_dev)


def filter_cost(para_in, t, y, x_dot, filter, cost):
        y_hat_dot = filter.filter_fun(t, y, para = para_in)
        return cost.cost(y_hat_dot, x_dot)

t, x, x_dot = sine.get_fun()
y = white.apply_noise(x)

y_hat2 = butter2.filter_fun(t, y)
y_hat3 = butter3.filter_fun(t, y)
y_hat4 = butter4.filter_fun(t, y)
y_hat5 = butter5.filter_fun(t, y)
standard_cost = cost.cost(y,x)
butter_cost2 = cost.cost(y_hat2,x)
butter_cost3 = cost.cost(y_hat3,x)
butter_cost4 = cost.cost(y_hat4,x)
butter_cost5 = cost.cost(y_hat5,x)


plot.plot_sig(t,[x,y,y_hat2,y_hat3,y_hat4,y_hat5],['sine','noisy sine (' + str(standard_cost) + ')',
'Butterworth2 (' + str(butter_cost2) + ')',
'Butterworth3 (' + str(butter_cost3) + ')',
'Butterworth4 (' + str(butter_cost4) + ')',
'Butterworth5 (' + str(butter_cost5) + ')',],True)
plt.show()

y_dot = np.diff(y, prepend=0)

y_dot_hat2 = butter2.filter_diff(t, y)
y_dot_hat3 = butter3.filter_diff(t, y)
y_dot_hat4 = butter4.filter_diff(t, y)
y_dot_hat5 = butter5.filter_diff(t, y)
standard_cost = cost.cost(y_dot,x_dot)
butter_cost2 = cost.cost(y_dot_hat2,x_dot)
butter_cost3 = cost.cost(y_dot_hat3,x_dot)
butter_cost4 = cost.cost(y_dot_hat4,x_dot)
butter_cost5 = cost.cost(y_dot_hat5,x_dot)

plot.plot_sig(t,[x_dot,y_dot,y_dot_hat2,y_dot_hat3,y_dot_hat4,y_dot_hat5],['cos','diff sin (' + str(standard_cost) + ')',
'Butterworth2 (' + str(butter_cost2) + ')',
'Butterworth3 (' + str(butter_cost3) + ')',
'Butterworth4 (' + str(butter_cost4) + ')',
'Butterworth5 (' + str(butter_cost5) + ')'],True)
plt.show()