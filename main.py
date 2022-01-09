from cost import Cost_Enum
from scipy.optimize import minimize

from input_function import *
from noises import *
from filter import *
from plot_sig import *
from cost import *

# true input functions
sine    = Input_Function(Input_Enum.SINE, [1,0.4,0,0])
polynom = Input_Function(Input_Enum.POLYNOM, [1,1,1,1]) #coefs in descending order 2x^2+1 = [2,0,1]
exp     = Input_Function(Input_Enum.EXP, [1,1.5,0,0]) #coefs [a,b,c,d]= a*e^(t/b+c)+d
# noise generation
noise_std_dev = 0.1
learn_noises  = [       [Noise(Noise_Enum.WHITE, noise_std_dev, seed=i) for i in range(10)], \
                        [Noise(Noise_Enum.PINK, noise_std_dev, seed=i) for i in range(10)], \
                        [Noise(Noise_Enum.QUANT, noise_std_dev, seed=i) for i in range(10)]      ]               # oder PINK, QUANT
test_noises   = [Noise(Noise_Enum.WHITE, noise_std_dev, seed=i) for i in range(10, 20)]

# filters
pt1     = Filter(Filter_Enum.PT1, 1e2)
wiener  = Filter(Filter_Enum.WIENER, noise_std_dev)
kalman  = Filter(Filter_Enum.KALMAN, parameters=None)

# cost objects and functions
cost    = Cost(Cost_Enum.MSE)                                                                                   # oder AVG_LOSS, PHASE_SHIFT
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
time, true_input, true_input_dot = polynom.get_fun()                                                               # oder polynom, exp
norm_freq = time[:round(len(time)/2)] / (time[-1] - time[0])
whitened_input = learn_noises[0][0].apply_noise(true_input)
pinked_input = learn_noises[1][0].apply_noise(true_input)
quanted_input = learn_noises[2][0].apply_noise(true_input)

# unfiltered differentiation of noisy signal
diff_finite = fwd_diff(time, whitened_input)

# PT1 smoothing
# optimize pt1 parameter
param_pt1 = minimize(filter_cost,0.1,args=(time, whitened_input, true_input, pt1, cost))
param_pt1 = param_pt1.x
# apply PT1
y_hat_pt1 = pt1.filter_fun(time, whitened_input, para = param_pt1)
# numeric differentiation
y_hat_dot_pt1 = fwd_diff(time, y_hat_pt1)
print(f"PT1 smoothing and numeric diff: \n \
        Optimal PT1 parameter: {param_pt1} \n \n")

# Wiener smoothing
# apply Wiener filter with known noise std dev
y_hat_wiener = wiener.filter_fun(time, whitened_input, [noise_std_dev, (whitened_input-true_input), 'freq'])
# get transfer function of Wiener filter
tf_wiener = y_hat_wiener[1]
# get time series of Wiener output
y_hat_wiener = y_hat_wiener[0]
# numeric differentiation
y_hat_dot_wiener = fwd_diff(time, y_hat_wiener)

# Kalman smoothing
# filter paramters
kalman_filter_order  = 2
kalman_process_noise = 2e4
x_start_guess        = np.array([[0], [2*np.pi*10]])
# optimize Kalman parameter
kalman_process_noise = minimize(kalman_filter_cost, kalman_process_noise, args=(time, whitened_input, [kalman_filter_order, x_start_guess, noise_std_dev], true_input, kalman, cost), method='Nelder-Mead')
kalman_process_noise = kalman_process_noise.x
# apply Kalman filter
y_hat_kalman = kalman.filter_fun(time, whitened_input, para=[kalman_filter_order, x_start_guess, noise_std_dev, kalman_process_noise])
# get differentiation from state
y_hat_dot_kalman = y_hat_kalman[1]
# get transfer function of Kalman filter
tf_kalman = y_hat_kalman[2]
# get time series of Kalman output
y_hat_kalman = y_hat_kalman[0]
print(f"Kalman smoothing: \n \
        Optimal Kalman parameter: {kalman_process_noise} \n \n")


print(f"Mean Squared Error of Differentials: \n \
        Forward Difference: {cost.cost(diff_finite, true_input_dot)} \n \
        PT1: {cost.cost(y_hat_dot_pt1, true_input_dot)} \n \
        Wiener: {cost.cost(y_hat_dot_wiener, true_input_dot)} \n \
        Kalman: {cost.cost(y_hat_dot_kalman, true_input_dot)} \n" )

# plot results
plot_sub.plot_sig(time, [true_input_dot, y_hat_dot_wiener], ["true diff sine", "diff Wiener"])
plot_sub.plot_sig(norm_freq, [tf_wiener, tf_kalman], ["Wiener TF", "Kalman TF"], time_domain=False)
plot_sub.plot_sig(time, [diff_finite, y_hat_dot_wiener], ["diff unsmoothed", "diff Wiener"])
plot_sub.plot_sig(time, [true_input, whitened_input, y_hat_wiener], ["true sine", "noisy sine", "Wiener smoothed"])
plot_sub.plot_sig(time, [whitened_input, y_hat_kalman, y_hat_dot_kalman], ["y", "Kalman", "dot Kalman"])
plt.show()