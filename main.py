import numpy as np

from environment import *
from smoothings import *

# sine signal and time series
tt_length = 1000
tt_step = 0.01
tt = np.arange(0, tt_length * tt_step, tt_step)
sine_period = 5
true_sine = np.sin(2*np.pi/sine_period * tt)
# generate white noise
mean = 0
std_dev = 0.05
noise = np.random.normal(mean, std_dev, tt_length)
# add noise to signal
noisy_sine = true_sine + noise


# differentiate noisy signal
diff_finite = fwd_diff(tt, noisy_sine)
# PT1-smoothing
diff_pt1_smoothed = fwd_diff(tt, pt1_smooth(tt, noisy_sine, 5))


# plot results
plot_sig(tt, [noisy_sine, true_sine, diff_finite, diff_pt1_smoothed], ["with noise", "no noise", "Vorwärtsdifferenz", "diff_pt1_smoothed"])
plt.show()