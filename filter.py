import numpy as np
from numpy.core.numeric import isscalar
from scipy.signal import lfilter, cont2discrete
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

    def __filter_fun_savgol(self,t,y,para): #kausaler savgol filter. point of evluation is end of window_length para=[m,polorder,diff=None]
        ###################################################
        #Compute H Matrix: Only Polyorder and window needed
        m=para[0]
        polyorder=para[1]
        window_length=2*m+1
        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length.")
        x_vector = np.arange(-m, window_length-m) #for a=H*x
        order = np.arange(polyorder + 1).reshape(-1, 1)
        A_t = x_vector ** order #for H=(A_trans*A)^-1*A_trans    
        A=np.transpose(A_t)
        B=A_t.dot(A)
        B_inv=np.linalg.inv(B)
        H=B_inv.dot(A_t)
        ###################################################
        #Filter
        y_hat=[]
        for i in range(0,len(y)-window_length+1):
            y_signal=np.transpose([y[0+i:window_length+i]])
            f=[]
            for l in range(0,polyorder+1):
                f.append(y[window_length+i-1]**l)
            
            f=np.asarray(f) 
            y_zwischen=f.dot(H.dot(y_signal))
            y_hat.append(y_zwischen[0])
        ###################################################
        return y_hat
        
        
        
        
        

    def __filter_fun_wiener(self,t,y,para):
        # Parameter: Noise standard Deviation
        noise_stdev = para
        S_nn = noise_stdev**2*np.ones(len(y))
        S_yy = np.square(np.abs(np.fft.fft(y)/len(y)))
        H_noncausal = np.maximum(0, np.divide(S_yy - S_nn , S_yy))
        Y_hat = np.multiply(H_noncausal, np.fft.fft(y))
        return [np.real(np.fft.ifft(Y_hat)), H_noncausal[:round(len(H_noncausal)/2)]]

    def __filter_fun_kalman(self,t,y,para):
        # Parameter: x_0 estimation, stdev of output noise, stdev of process noise
        T_sample = (t[-1] - t[0]) / (len(t) - 1)

        # initialization
        x_est = para[0]
        output_n__stdev = para[1]
        process_n_stdev = para[2]
        filter_order = para[3]

        A = np.zeros((filter_order, filter_order))
        for i in range(0, filter_order-1):
            A[i, i+1] = 1
        B = np.zeros((filter_order, 1))
        B[-1] = 1
        C = np.zeros((1, filter_order))
        C[0, 0] = 1
        D = np.array([[0]])

        disc_sys = cont2discrete((A,B,C,D), T_sample)
        F = disc_sys[0]
        B = disc_sys[1]
        H = disc_sys[2]
        Q = B * B.transpose() * process_n_stdev**2
        R = np.array([output_n__stdev**2]).reshape(1, 1)
        P = 1e-3*np.eye(filter_order)

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

        # transfer function
        G_tf = np.maximum(0, np.divide(abs(np.fft.fft(x_1_list)), abs(np.fft.fft(y))))

        return [x_1_list, x_2_list, G_tf[:round(len(G_tf)/2)]]
        

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