from scipy.signal import lfilter

# Smoothing filters
def pt1_smooth(tt, signal, cutoff_freq):
    T_sample = (tt[-1] - tt[0]) / (len(tt) - 1)
    num_coeff = [T_sample * cutoff_freq]
    den_coeff = [1, T_sample * cutoff_freq - 1]
    smoothed = lfilter(num_coeff, den_coeff, signal)
    return smoothed