import numpy as np
from scipy.signal import lfilter, wiener, convolve, correlate
from scipy.linalg import toeplitz
from adjusted_scipy_savgol_filter import *
from enum import Enum

class Filter_Enum(Enum):
    PT1             = 'PT1-Filter'
    SAVGOL          = 'Savitzky Golay Filter'
    WIENER          = 'Wiener Filter'
    DIFF_QUOTIENT   = 'Gleitender Differenzenquotient'
    BROWN_HOLT      ='Lineare Exponentielle Glättung'
    BUTTERWORTH     ='Butterworth Filter'
    CHEBYCHEV       ='Chebychev filter'
    ROB_EX_DIFF     ='Robust Exact Differentiator'
    KALMAN          ='Kalman'


class Filter:
    type                = Filter_Enum.SAVGOL

    def __init__(self,type,parameters):
        self.type               = type
        self.parameters         = parameters

    def filter_fun(self,t,y,para=None):
        para = self.parameters if para is None else para
        match self.type:
            case Filter_Enum.PT1:
                return self.__filter_fun_pt1(t,y,para)
            case Filter_Enum.SAVGOL:
                return self.__filter_fun_savgol(t,y,para)
            case Filter_Enum.WIENER:
                return self.__filter_fun_wiener(t,y,para)
            case Filter_Enum.DIFF_QUOTIENT:
                return self.__filter_fun_diff(t,y,para)
            case Filter_Enum.BROWN_HOLT:
                return self.__filter_fun_brownholt(t,y,para)
            case Filter_Enum.BUTTERWORTH:
                return self.__filter_fun_butterworth(t,y,para)
            case Filter_Enum.CHEBYCHEV:
                return self.__filter_fun_chebychev(t,y,para)
            case Filter_Enum.ROB_EX_DIFF:
                return self.__filter_fun_robexdiff(t,y,para)
                
    def filter_diff(self,t,y,para=None):
        para = self.parameters if para is None else para
        match self.type:
            case Filter_Enum.PT1:
                return self.__filter_diff_pt1(t,y,para)
            case Filter_Enum.SAVGOL:
                return self.__filter_diff_savgol(t,y,para)
            case Filter_Enum.WIENER:
                return self.__filter_diff_wiener(t,y,para)
            case Filter_Enum.DIFF_QUOTIENT:
                return self.__filter_diff_diff(t,y,para)
            case Filter_Enum.BROWN_HOLT:
                return self.__filter_diff_brownholt(t,y,para)
            case Filter_Enum.BUTTERWORTH:
                return self.__filter_diff_butterworth(t,y,para)
            case Filter_Enum.CHEBYCHEV:
                return self.__filter_diff_chebychev(t,y,para)
            case Filter_Enum.ROB_EX_DIFF:
                return self.__filter_diff_robexdiff(t,y,para)

####################################################################
####################### Filter Functions ###########################
####################################################################


    def __filter_fun_pt1(self,t,y,para):
        # Parameter: Cutoff frequency
        T_sample = (t[-1] - t[0]) / (len(t) - 1)
        num_coeff = [T_sample * para]
        den_coeff = [1, T_sample * para - 1]
        return lfilter(num_coeff, den_coeff, y)

    def __filter_fun_savgol(self,t,y,para):
        return y
    def __filter_fun_wiener(self,t,y,para):
        # Parameter: Noise standard Deviation
        S_nn = para**2*np.ones(len(y))
        S_yy = np.square(np.abs(np.fft.fft(y)/len(y)))
        H_noncausal = np.maximum(0, np.divide(S_yy - S_nn , S_yy))
        Y_hat = np.multiply(H_noncausal, np.fft.fft(y))
        return np.real(np.fft.ifft(Y_hat))
        
    def __filter_fun_diff(self,t,y,para):
        return y
    def __filter_fun_brownholt(self,t,y,para):
        return y
    def __filter_fun_butterworth(self,t,y,para):
        return y
    def __filter_fun_chebychev(self,t,y,para):
        return y
    def __filter_fun_robexdiff(self,t,y,para):
        return y
    

####################################################################
####################### Diff Functions #############################
####################################################################

    def __filter_diff_pt1(self,t,y,para):
            # Parameter: Cutoff frequency
        T_sample = (t[-1] - t[0]) / (len(t) - 1)
        num_coeff = [T_sample * para]
        den_coeff = [1, T_sample * para - 1]
        return lfilter(num_coeff, den_coeff, y)

    def __filter_diff_savgol(self,t,y,para):
        return y
    def __filter_diff_wiener(self,t,y,para):
        # Parameter: Noise standard Deviation
        S_nn = para**2*np.ones(len(y))
        S_yy = np.square(np.abs(np.fft.fft(y)/len(y)))
        H_noncausal = np.maximum(0, np.divide(S_yy - S_nn , S_yy))
        Y_hat = np.multiply(H_noncausal, np.fft.fft(y))
        return np.real(np.fft.ifft(Y_hat))
        
    def __filter_diff_diff(self,t,y,para):
        return y
    def __filter_diff_brownholt(self,t,y,para):
        return y
    def __filter_diff_butterworth(self,t,y,para):
        return y
    def __filter_diff_chebychev(self,t,y,para):
        return y
    def __filter_diff_robexdiff(self,t,y,para):
        return y

####################################################################
########################## Old Stuff ###############################
####################################################################

# https://github.com/zziz/kalman-filter
class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    # prediction step
    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    # correction / update step
    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def kalman_smooth(tt, signal, noise_stdev, process_stdev):
    T_sample = (tt[-1] - tt[0]) / (len(tt) - 1)
    F = np.array([[1, T_sample], [0, 1]])
    B = np.array([0, 0]).reshape(2, 1)#np.array([T_sample**2/2, T_sample]).reshape(2, 1)
    H = np.array([[1, 0]]).reshape(1, 2)
    Q = np.array([[process_stdev, 0], [0, process_stdev]])
    R = np.array([noise_stdev**2]).reshape(1, 1)

    measurements = signal
    kf = KalmanFilter(F = F, B = B, H = H, Q = Q, R = R)
    predictions = []
    x_1 = []
    x_2 = []

    for z in measurements:
        x_hat = kf.predict()
        predictions.append(np.dot(H, x_hat))
        x_1.append(x_hat[0])
        x_2.append(x_hat[1])
        kf.update(z)
    
    return [x_1, x_2]

def savgol_smooth(signal,window_length,polyorder,sample_frequency,pos=None,deriv=0):
    #general savgol filtering in time domain
    winl=window_length
    savgol_signal_time=adjusted_savgol_filter(signal,winl,polyorder,pos=pos,deriv=deriv)


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