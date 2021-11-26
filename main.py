from cost import Cost_Enum
from scipy.optimize import minimize

from input_function import *
from noises import *
from filter import *
from plot_sig import *
from cost import *

noise_std_dev   = 0.1
alpha           = 0.4


sine    = Input_Function(Input_Enum.SINE, [1, 0.1, 0, 0.5])
white   = Noise(Noise_Enum.WHITE, noise_std_dev)
pt1     = Filter(Filter_Enum.PT1, 1e2)
wiener  = Filter(Filter_Enum.WIENER, noise_std_dev)
kalman  = Filter(Filter_Enum.KALMAN, noise_std_dev)
brown   = Filter(Filter_Enum.BROWN_HOLT, alpha)
plot    = Plot_Sig(Plot_Enum.MULTI, "Overview", [])
plot_sub= Plot_Sig(Plot_Enum.SUBPLOT, "Overview", [])
cost    = Cost(Cost_Enum.MSE)
plot_s  = Plot_Sig(Plot_Enum.SLIDER, "Detailed View with Slider",[])
savgol  = Filter(Filter_Enum.SAVGOL, parameters=None)

time, true_sine, true_sine_dot = sine.get_fun()
y = white.apply_noise(true_sine)

def filter_cost(para_in, t, y, n, filter, cost):
        y_hat_pt1 = filter.filter_fun(t, y, para = para_in)
        return cost.cost(y_hat_pt1, n)



# differentiate noisy signal
diff_finite = fwd_diff(time, y)
# PT1-smoothing
f_min = minimize(filter_cost,0.1,args=(time, y, true_sine, pt1,cost))
y_hat_pt1 = pt1.filter_fun(time, y, para = f_min.x)
y_hat_dot_pt1 = fwd_diff(time, y_hat_pt1)
f_min = minimize(filter_cost,0.1,args=(time, y_hat_dot_pt1, true_sine_dot, pt1,cost))
y_hat_dot_pt1 = pt1.filter_fun(time, y_hat_dot_pt1, para = f_min.x)


# Wiener-smoothing
# Frage: wie noise std_dev am besten schaetzen, wenn unbekannt?
y_hat_wiener = wiener.filter_fun(time, y, noise_std_dev)
y_hat_dot_wiener = fwd_diff(time, y_hat_wiener)
# Kalman-smoothing
y_hat_kalman = kalman_smooth(time, y, noise_std_dev, 0.3*noise_std_dev)
y_hat_dot_kalman = y_hat_kalman[1]
y_hat_kalman = y_hat_kalman[0]

# Brown Holt
alpha_min = minimize(filter_cost,alpha,args=(time, y, true_sine, brown ,cost))
y_hat_brown = brown.filter_fun(time, y, para = alpha_min.x)
y_hat_dot_brown = fwd_diff(time, y_hat_brown)

print(f"Mean Squared Error of Differentials: \n \
        Forward Difference: {cost.cost(diff_finite, true_sine_dot)} \n \
        PT1: {cost.cost(y_hat_dot_pt1, true_sine_dot)} \n \
        Wiener: {cost.cost(y_hat_dot_wiener, true_sine_dot)} \n \
        Kalman: {cost.cost(y_hat_dot_kalman, true_sine_dot)} \n \
        Brown-Holt: {cost.cost(y_hat_dot_brown, true_sine_dot)} \n" )

# plot results
plot.plot_sig(time, [true_sine_dot, y_hat_dot_wiener, y_hat_dot_pt1, y_hat_dot_kalman, y_hat_dot_brown], ["true diff sine", "diff Wiener", "diff PT1", "diff Kalman", "diff Brown"])
plot.plot_sig(time, [diff_finite, y_hat_dot_wiener, y_hat_dot_pt1, y_hat_dot_kalman, y_hat_dot_brown], ["diff unsmoothed", "diff Wiener", "diff PT1", "diff Kalman", "diff Brown"])
plot.plot_sig(time, [true_sine, y, y_hat_pt1, y_hat_wiener, y_hat_kalman, y_hat_brown], ["true sine", "noisy sine", "PT1 smoothed", "Wiener smoothed", "Kalman smoothed", "Brown smoothed"])

# plot_sub.plot_sig(time,[true_sine_dot, y_hat_dot_wiener, y_hat_dot_pt1, y_hat_dot_kalman, y_hat_dot_brown], ["true diff sine", "diff Wiener", "diff PT1", "diff Kalman", "diff Brown"])
# plot_sub.plot_sig(time,[true_sine, y_hat_wiener, y_hat_pt1, y_hat_kalman, y_hat_brown], ["true diff sine", "diff Wiener", "diff PT1", "diff Kalman", "diff Brown"])
# 
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