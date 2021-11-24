from cost import Cost_Enum
from scipy.optimize import minimize

from input_function import *
from noises import *
from filter import *
from plot_sig import *
from cost import *

noise_std_dev = 0.1

sine    = Input_Function(Input_Enum.SINE, [1, 0.1, 0, 0.5])
white   = Noise(Noise_Enum.WHITE, noise_std_dev)
pt1     = Filter(Filter_Enum.PT1, 1e2)
wiener  = Filter(Filter_Enum.WIENER, noise_std_dev)
kalman  = Filter(Filter_Enum.KALMAN, [0, 10])
plot    = Plot_Sig(Plot_Enum.MULTI, "Overview", [])
cost    = Cost(Cost_Enum.MSE)
plot_s  = Plot_Sig(Plot_Enum.SLIDER, "Detailed View with Slider",[])
savgol  = Filter(Filter_Enum.SAVGOL, parameters=None)

time, true_sine, true_sine_dot = sine.get_fun()
y = white.apply_noise(true_sine)

def filter_cost(para_in, t, y, n, filter, cost):
        y_hat_pt1 = filter.filter_fun(t, y, para = para_in)
        return cost.cost(y_hat_pt1, n)

f_min = minimize(filter_cost,0.1,args=(time, y, true_sine, pt1,cost))
print("Minimal solution found. f_min = ")
print(f_min.x)
y_hat_pt1 = pt1.filter_fun(time, y, para = f_min.x)
print(cost.cost(y, true_sine))
print(cost.cost(y_hat_pt1, true_sine))
#plot.plot_sig(time, [true_sine, y, y_hat_pt1],["Roh","Rausch", "Filter"])

# differentiate noisy signal
diff_finite = fwd_diff(time, y)
# PT1-smoothing
y_hat_dot_pt1 = fwd_diff(time, y_hat_pt1)
# Wiener-smoothing
# Frage: wie noise std_dev am besten schaetzen, wenn unbekannt?
y_hat_wiener = wiener.filter_fun(time, y, noise_std_dev)
y_hat_dot_wiener = fwd_diff(time, y_hat_wiener)


# Kalman-smoothing
y_kalman = 10*time  
y_kalman = white.apply_noise(y_kalman)
y_hat_kalman = kalman.filter_fun(time, y_kalman, para=[[0, 10]])
y_hat_dot_kalman = y_hat_kalman[1]
y_hat_kalman = y_hat_kalman[0]

print(f"Mean Squared Error of Differentials: \n \
        Forward Difference: {cost.cost(diff_finite, true_sine_dot)} \n \
        PT1: {cost.cost(y_hat_dot_pt1, true_sine_dot)} \n \
        Wiener: {cost.cost(y_hat_dot_wiener, true_sine_dot)} \n \
        Kalman: {cost.cost(y_hat_dot_kalman, true_sine_dot)} \n" )

# plot results
#plot.plot_sig(time, [true_sine_dot, y_hat_dot_wiener, y_hat_dot_pt1, y_hat_dot_kalman], ["true diff sine", "diff Wiener", "diff PT1", "diff Kalman"])
#plot.plot_sig(time, [diff_finite, y_hat_dot_wiener, y_hat_dot_pt1, y_hat_dot_kalman], ["diff unsmoothed", "diff Wiener", "diff PT1", "diff Kalman"])
#plot.plot_sig(time, [true_sine, y, y_hat_pt1, y_hat_wiener, y_hat_kalman], ["true sine", "noisy sine", "PT1 smoothed", "Wiener smoothed", "Kalman smoothed"])
plot.plot_sig(time, [y_kalman, y_hat_kalman, y_hat_dot_kalman], ["true", "Kalman", "dot Kalman"])
plt.show()


'''
#test of savgol_smooth funktioniert nur ueber umwege wegen para, aber funktioniert
x=np.linspace(0,2*np.pi,100)
y=np.sin(x)+np.cos(x)+np.random.random(100)
#y_filtered,_,_=savgol_smooth(y,99,3,160,deriv=0)
y_hat_savgol=savgol.filter_fun(x,y,para=[3, 99])
plot_s.plot_slider(x,[y, y_hat_savgol],['noisy sine','savgol smoothed'],[3,99],savgol)
#plt.show()
'''