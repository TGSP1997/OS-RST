import numpy as np

# generate sine input function and its derivative
def sine_input(sampling_period, point_count, sine_period):
    time = np.arange(0, point_count * sampling_period, sampling_period)
    sine = np.sin(2*np.pi/sine_period * time)
    diff_sine = np.cos(2*np.pi/sine_period * time)

    return [sine, diff_sine]