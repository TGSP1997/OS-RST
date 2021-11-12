import numpy as np
from scipy.signal import lfilter, wiener, convolve, correlate
from scipy.linalg import toeplitz
from adjusted_scipy_savgol_filter import *

# Smoothing filters
def pt1_smooth(tt, signal, cutoff_freq):
    T_sample = (tt[-1] - tt[0]) / (len(tt) - 1)
    num_coeff = [T_sample * cutoff_freq]
    den_coeff = [1, T_sample * cutoff_freq - 1]
    smoothed = lfilter(num_coeff, den_coeff, signal)
    return smoothed

# noncausal FIR Wiener filter
def wiener_smooth(signal, noise, filter_order, noise_stdev):
    S_nn = np.square(np.abs(np.fft.fft(noise)/len(noise)))
    S_yy = np.square(np.abs(np.fft.fft(signal)/len(signal)))
    H_noncausal = np.maximum(0, np.divide(S_yy - S_nn , S_yy))
    X_hat = np.multiply(H_noncausal, np.fft.fft(signal))
    x_hat = np.real(np.fft.ifft(X_hat))
    return x_hat

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

def savgol_smooth(signal,window_length,polyorder,sample_frequency,pos=None):
    #general savgol filtering in time domain
    winl=window_length
    savgol_signal_time=adjusted_savgol_filter(signal,winl,polyorder,pos=pos)


    #general savgol filtering in frequency domain
    '''
    sources: 
    https://github.com/chtaal/pydata2017/blob/master/scripts/gen_presentation_figs.py
    https://www.youtube.com/watch?v=0TSvo2hOKo0

    '''
    fs=sample_frequency
    '''
    https://cran.r-project.org/web/packages/signal/signal.pdf
    Selecting an FFT length greater than the window length
    does not add any information to the spectrum, but it is a good way to interpolate between frequency
    points which can make for prettier spectrograms
    '''
    fft_length=winl*30 
    #calculate savgol coeeficients
    fft_coeffs = adjusted_savgol_coeffs(window_length=winl, polyorder=polyorder, pos=pos)
    convolved_signal=convolve(signal,fft_coeffs)

    #Y = 2*np.abs(np.fft.rfft(convolved_signal, fft_length))/winl
    savgol_signal_freq = np.fft.rfft(convolved_signal, fft_length)
    freq_axes = np.fft.rfftfreq(fft_length, d=1 / fs)

    return savgol_signal_time, savgol_signal_freq, freq_axes

# numerical differentiation
def fwd_diff(tt, signal):
    diff = np.zeros(len(tt))
    for i in range(0, len(tt)-1):
        diff[i] = (signal[i+1] - signal[i]) / (tt[1] - tt[0])
    return diff