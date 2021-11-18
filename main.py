import numpy as np
from cost import Cost_Enum
from scipy.optimize import minimize

from input_function import *
from noises import *
from filter import *
from plot_sig import *
from cost import *

sine    = Input_Function(Input_Enum.SINE,[1,0.1,0,0.5])
white   = Noise(Noise_Enum.WHITE,0.4)
pt1     = Filter(Filter_Enum.PT1,1e2)
plot    = Plot_Sig(Plot_Enum.MULTI,"Overview",[])
cost    = Cost(Cost_Enum.MSE)


t,n,n_dot = sine.get_fun()
y = white.apply_noise(n)

def filter_cost(para_in,t,y,n,filter,cost):
        y_hat = filter.filter_fun(t,y,para = para_in)
        return cost.cost(y_hat,n)

f_min = minimize(filter_cost,0.1,args=(t,y,n,pt1,cost))
print("Minimal solution found. f_min = ")
print(f_min.x)
y_hat = pt1.filter_fun(t,y,para = f_min.x)
print(cost.cost(y,n))
print(cost.cost(y_hat,n))
plot.plot_sig(t,[n,y,y_hat],["Roh","Rausch", "Filter"])
'''
# time array and sine signal
tt_length = 1000
tt_step = 0.01
tt = np.arange(0, tt_length * tt_step, tt_step)
sine_period = 5
true_sine, true_diff_sine = sine_input(tt_step, tt_length, sine_period)

# add noise to signal
mean = 0 
std_dev = 0.05
rand_seed = 0
noise = white_noise(mean, std_dev, tt_length, rand_seed)
noisy_sine = true_sine + noise

# differentiate noisy signal
diff_finite = fwd_diff(tt, noisy_sine)
# PT1-smoothing
pt1_smoothed = pt1_smooth(tt, noisy_sine, 5)
diff_pt1_smoothed = fwd_diff(tt, pt1_smoothed)
# Wiener-smoothing
# Frage: wie noise std_dev am besten schaetzen, wenn unbekannt?
wiener_smoothed = wiener_smooth(noisy_sine, std_dev)
diff_wiener_smoothed = fwd_diff(tt, wiener_smoothed)
# Kalman-smoothing
kalman_smoothed = kalman_smooth(tt, noisy_sine, std_dev, 0.3*std_dev)
diff_kalman_smoothed = kalman_smoothed[1]
kalman_smoothed = kalman_smoothed[0]

print(f"Mean Squared Error of Differentials: \n \
        Forward Difference: {mean_squared_error(diff_finite, true_diff_sine)} \n \
        PT1: {mean_squared_error(diff_pt1_smoothed, true_diff_sine)} \n \
        Wiener: {mean_squared_error(diff_wiener_smoothed, true_diff_sine)} \n \
        Kalman: {mean_squared_error(diff_kalman_smoothed, true_diff_sine)} \n" )

# plot results
plot_time_sig(tt, [true_diff_sine, diff_wiener_smoothed, diff_pt1_smoothed, diff_kalman_smoothed], ["true diff sine", "diff Wiener", "diff PT1", "diff Kalman"])
plot_time_sig(tt, [diff_finite, diff_wiener_smoothed, diff_pt1_smoothed, diff_kalman_smoothed], ["diff unsmoothed", "diff Wiener", "diff PT1", "diff Kalman"])
plot_time_sig(tt, [true_sine, noisy_sine, pt1_smoothed, wiener_smoothed, kalman_smoothed], ["true sine", "noisy sine", "PT1 smoothed", "Wiener smoothed", "Kalman smoothed"])
plt.show()



#test of savgol_smooth
x=np.linspace(0,2*np.pi,100)
y=np.sin(x)+np.cos(x)+np.random.random(100)
y_filtered,_,_=savgol_smooth(y,99,3,160,deriv=0)
plot_time_sig(x,[y,y_filtered],['unfiltered','filtered'])
plt.show()
'''