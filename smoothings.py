from scipy.signal import lti, lsim

# Smoothing filters
def pt1_smooth(tt, signal, cutoff_freq):
    num_coeff = [1]
    den_coeff = [1/cutoff_freq, 1]
    trans_func = lti(num_coeff, den_coeff)
    diff = lsim(trans_func, signal, tt)[1]
    return diff