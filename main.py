from cost import Cost_Enum
from scipy.optimize import minimize

from input_function import *
from noises import *
from filter import *
from plot_sig import *
from cost import *

# true input functions
sine    = Input_Function(Input_Enum.SINE, [1, 0.1, 0, 0])
polynom = Input_Function(Input_Enum.POLYNOM, [1,2,3,4]) #coefs in descending order 2x^2+1 = [2,0,1]
exp     = Input_Function(Input_Enum.EXP, [1,2,0,0]) #coefs [a,b,c,d]= a*e^(t/b+c)+d
# noise generation
noise_std_dev   = 0.1
white   = Noise(Noise_Enum.WHITE, noise_std_dev)

# filters
pt1     = Filter(Filter_Enum.PT1, 1e2)
wiener  = Filter(Filter_Enum.WIENER, noise_std_dev)
kalman  = Filter(Filter_Enum.KALMAN, parameters=None)

# cost objects and functions
cost    = Cost(Cost_Enum.MSE)
def filter_cost(para_in, t, y, true_y, filter, cost):
        y_hat_pt1 = filter.filter_fun(t, y, para = para_in)
        return cost.cost(y_hat_pt1, true_y)
def kalman_filter_cost(para_in, t, y, para_filt, true_y, filter, cost):
        y_hat_kalman = filter.filter_fun(t, y, para = [para_filt[0], para_filt[1], para_filt[2], para_in])[0]
        return cost.cost(y_hat_kalman, true_y)

# plot objects
plot    = Plot_Sig(Plot_Enum.MULTI, "Overview", [])
plot_sub= Plot_Sig(Plot_Enum.SUBPLOT, "Overview", [])

# apply noise and generate time series
time, true_sine, true_sine_dot = sine.get_fun()
norm_freq = time[:round(len(time)/2)] / (time[-1] - time[0])
y = white.apply_noise(true_sine)

# optimize pt1 parameter
param_pt1 = minimize(filter_cost,0.1,args=(time, y, true_sine, pt1,cost))
y_hat_pt1 = pt1.filter_fun(time, y, para = param_pt1.x)
# differentiate noisy signal
diff_finite = fwd_diff(time, y)
# pt1 smoothing
y_hat_dot_pt1 = fwd_diff(time, y_hat_pt1)
param_pt1 = minimize(filter_cost,0.1,args=(time, y_hat_dot_pt1, true_sine_dot, pt1,cost))
y_hat_dot_pt1 = pt1.filter_fun(time, y_hat_dot_pt1, para = param_pt1.x)

print("Minimal solution found. param_pt1 = ")
print(param_pt1.x)
print(cost.cost(y, true_sine))
print(cost.cost(y_hat_pt1, true_sine))
plot.plot_sig(time, [true_sine, y, y_hat_pt1],["Roh","Rausch", "Filter"])


# Wiener-smoothing
# Frage: wie noise std_dev am besten schaetzen, wenn unbekannt?
y_hat_wiener = wiener.filter_fun(time, y, noise_std_dev)
tf_wiener = y_hat_wiener[1]
y_hat_wiener = y_hat_wiener[0]
y_hat_dot_wiener = fwd_diff(time, y_hat_wiener)


# Kalman-smoothing
kalman_filter_order = 2
process_std_dev = 2e4
x_start_guess = np.array([[0], [2*np.pi*10]])
kalman_process_noise = minimize(kalman_filter_cost, process_std_dev, args=(time, y, [kalman_filter_order, x_start_guess, noise_std_dev], true_sine, kalman, cost), method='Nelder-Mead')
kalman_process_noise = kalman_process_noise.x
print(kalman_process_noise)
y_hat_kalman = kalman.filter_fun(time, y, para=[kalman_filter_order, x_start_guess, noise_std_dev, kalman_process_noise])
y_hat_dot_kalman = y_hat_kalman[1]
tf_kalman = y_hat_kalman[2]
y_hat_kalman = y_hat_kalman[0]


print(f"Mean Squared Error of Differentials: \n \
        Forward Difference: {cost.cost(diff_finite, true_sine_dot)} \n \
        PT1: {cost.cost(y_hat_dot_pt1, true_sine_dot)} \n \
        Wiener: {cost.cost(y_hat_dot_wiener, true_sine_dot)} \n \
        Kalman: {cost.cost(y_hat_dot_kalman, true_sine_dot)} \n" )

# plot results
plot.plot_sig(time, [true_sine_dot, y_hat_dot_wiener], ["true diff sine", "diff Wiener"])
plot.plot_sig(norm_freq, [tf_wiener, tf_kalman], ["Wiener TF", "Kalman TF"], time_domain=False)
plot.plot_sig(time, [diff_finite, y_hat_dot_wiener], ["diff unsmoothed", "diff Wiener"])
plot.plot_sig(time, [true_sine, y, y_hat_wiener], ["true sine", "noisy sine", "Wiener smoothed"])
plot.plot_sig(time, [y, y_hat_kalman, y_hat_dot_kalman], ["y", "Kalman", "dot Kalman"])
plt.show()