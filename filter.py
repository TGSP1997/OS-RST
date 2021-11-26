import numpy as np
from numpy.core.numeric import isscalar
from scipy.signal import lfilter, wiener, convolve
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
            case Filter_Enum.KALMAN:
                return self.__filter_fun_kalman(t,y,para)
                
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
            case Filter_Enum.KALMAN:
                return self.__filter_diff_kalman(t,y,para)

####################################################################
####################### Filter Functions ###########################
####################################################################


    def __filter_fun_pt1(self,t,y,para):
        para = para if isscalar(para) else para[0] # Aus Gründen(TM) wandelt scp.minimize Skalare Werte in Vektoren mit Länge 1 um.
        # Parameter: Cutoff frequency
        T_sample = (t[-1] - t[0]) / (len(t) - 1)
        num_coeff = [T_sample * para]            
        den_coeff = [1, T_sample * para - 1]
        return lfilter(num_coeff, den_coeff, y)

    def __filter_fun_savgol(self,t,y,para): #wird noch ummgebaut, nach beratung mit betreuer
        savgol_signal_time=adjusted_savgol_filter(y,para[1],para[0])
        return savgol_signal_time

    def __filter_fun_wiener(self,t,y,para):
        # Parameter: Noise standard Deviation
        noise_stdev = para
        S_nn = noise_stdev**2*np.ones(len(y))
        S_yy = np.square(np.abs(np.fft.fft(y)/len(y)))
        H_noncausal = np.maximum(0, np.divide(S_yy - S_nn , S_yy))
        Y_hat = np.multiply(H_noncausal, np.fft.fft(y))
        return np.real(np.fft.ifft(Y_hat))
    
    def __filter_fun_kalman(self,t,y,para):
        # Parameter: x_0 estimation, stdev of output noise, stdev of process noise
        T_sample = (t[-1] - t[0]) / (len(t) - 1)

        # initialization
        x_est = para[0]
        output_n__stdev = para[1]
        process_n_stdev = para[2]

        F = np.array([[1, T_sample], [0, 1]])
        B = np.array([0.5*T_sample**2, T_sample]).reshape(2, 1)
        H = np.array([[1, 0]]).reshape(1, 2)
        Q = np.array([[0.25*T_sample**4, 0.5*T_sample**3], [0.5*T_sample**3, T_sample**2]])*process_n_stdev
        R = np.array([output_n__stdev**2]).reshape(1, 1)
        P = np.array([[1e-3, 0],[0, 1e-3]])

        x_1_list = x_est[0]
        x_2_list = x_est[1]
        k = 1
        while k < len(y):
            x_est_prior = F@x_est
            P_prior = H@P@H.transpose() + Q
            K = P_prior@H.transpose() @ np.linalg.inv(H@P_prior@H.transpose() + R)
            x_est = x_est_prior + K@(y[k] - H@x_est_prior)
            P = (np.eye(len(B)) - K@H)@P_prior
            x_1_list = np.append(x_1_list, x_est[0])
            x_2_list = np.append(x_2_list, x_est[1])
            k = k + 1
        return [x_1_list, x_2_list]

    def __filter_fun_diff(self,t,y,para):
        return y
    def __filter_fun_brownholt(self,t,y,para):
        # Parameter: Alpha
        y_hat = np.zeros(len(y))
        for i in range(1,len(y)):
            y_hat[i] = para*y[i] + (1-para)*y_hat[i-1]
        return y_hat
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
        para = para if isscalar(para) else para[0] # Aus Gründen(TM) wandelt scp.minimize Skalare Werte in Vektoren mit Länge 1 um.
            # Parameter: Cutoff frequency
        T_sample = (t[-1] - t[0]) / (len(t) - 1)
        num_coeff = [T_sample * para]
        den_coeff = [1, T_sample * para - 1]
        return lfilter(num_coeff, den_coeff, y)

    def __filter_diff_savgol(self,t,y,para):
        return y
    
    def __filter_diff_wiener(self,t,y,para):
        return y

    def __filter_diff_kalman(self,t,y,para):
        return y

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

# numerical differentiation
def fwd_diff(tt, signal):
    diff = np.zeros(len(tt))
    for i in range(0, len(tt)-1):
        diff[i] = (signal[i+1] - signal[i]) / (tt[1] - tt[0])
    return diff