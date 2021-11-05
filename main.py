import numpy as np

from environment import *
from smoothings import *

# sine signal and time series
tt_length = 1000
tt_step = 0.01
tt = np.arange(0, tt_length * tt_step, tt_step)
sine_period = 5
true_sine = np.sin(2*np.pi/sine_period * tt)
true_diff_sine = np.cos(2*np.pi/sine_period * tt)
# generate white noise
mean = 0
std_dev = 0.05
noise = np.random.normal(mean, std_dev, tt_length)
# add noise to signal
noisy_sine = true_sine + noise

# differentiate noisy signal
diff_finite = fwd_diff(tt, noisy_sine)
# PT1-smoothing
pt1_smoothed = pt1_smooth(tt, noisy_sine, 5)
diff_pt1_smoothed = fwd_diff(tt, pt1_smoothed)
# Wiener-smoothing
# Frage: wie noise std_dev am besten schaetzen, wenn unbekannt?
wiener_smoothed = wiener_smooth(noisy_sine, 49, std_dev)
diff_wiener_smoothed = fwd_diff(tt, wiener_smoothed)
# Kalman-smoothing
kalman_smoothed = kalman_smooth(noisy_sine, std_dev, 30*std_dev)
diff_kalman_smoothed = fwd_diff(tt, kalman_smoothed)

# plot results
plot_sig(tt, [true_diff_sine, diff_wiener_smoothed, diff_pt1_smoothed, diff_kalman_smoothed], ["true diff sine", "diff Wiener", "diff PT1", "diff Kalman"])
plot_sig(tt, [diff_finite, diff_wiener_smoothed, diff_pt1_smoothed, diff_kalman_smoothed], ["diff unsmoothed", "diff Wiener", "diff PT1", "diff Kalman"])
plot_sig(tt, [true_sine, noisy_sine, pt1_smoothed, wiener_smoothed, kalman_smoothed], ["true sine", "noisy sine", "PT1 smoothed", "Wiener smoothed", "Kalman smoothed"])
plt.show()