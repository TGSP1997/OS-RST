import numpy as np
from scipy.signal import lfilter, wiener

# Smoothing filters
def pt1_smooth(tt, signal, cutoff_freq):
    T_sample = (tt[-1] - tt[0]) / (len(tt) - 1)
    num_coeff = [T_sample * cutoff_freq]
    den_coeff = [1, T_sample * cutoff_freq - 1]
    smoothed = lfilter(num_coeff, den_coeff, signal)
    return smoothed

def wiener_smooth(signal, filter_order, noise_stdev):
    # scipy Implementation: https://tmramalho.github.io/blog/2013/04/09/an-introduction-to-smoothing-time-series-in-python-part-ii-wiener-filter-and-smoothing-splines/
    smoothed = wiener(signal, mysize=filter_order, noise=noise_stdev)
    return smoothed

# https://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html
def kalman_smooth(signal, noise_stdev, measure_stdev):
    n_iter = len(signal)
    sz = (n_iter,) # size of array
    Q = noise_stdev**2 # process variance
    R = measure_stdev**2 # estimate of measurement variance, change to see effect

    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor

    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0

    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q

        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(signal[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]

    return xhat