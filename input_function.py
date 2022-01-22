import numpy as np
import sympy as sp
from enum import Enum

class Input_Enum(Enum):
    SINE="sine"
    POLYNOM="polynom"
    EXP="exponential"

class Input_Function:
    type                = Input_Enum.SINE
    parameters          = []
    point_counter       = 1000
    sampling_period     = 1e-3

    def __init__(self,type,parameters,point_counter = 1000, sampling_period=1e-3):
        self.type               = type
        self.parameters         = parameters
        self.point_counter      = point_counter
        self.sampling_period    = sampling_period


    # Returns t, n, n_dot
    def get_fun(self):  
        match self.type:
            case Input_Enum.SINE:
                return self.__get_fun_sine()
            case Input_Enum.POLYNOM:
                return self.__get_fun_polynom()
            case Input_Enum.EXP:
                return self.__get_fun_exp()

# Ab hier bearbeiten

    def __get_fun_sine(self):
        t = np.arange(0, self.point_counter * self.sampling_period, self.sampling_period) #same as sine
        n = self.parameters[0]*np.sin(2*np.pi/self.parameters[1] * t + self.parameters[2]) + self.parameters[3]
        n_dot = 2*np.pi/self.parameters[1]*self.parameters[0]*np.cos(2*np.pi/self.parameters[1] * t + self.parameters[2])
        return t, n, n_dot

        
    def __get_fun_polynom(self):#coeefs in descending order 2x^2+1 = [2,0,1]
        t = np.arange(0, self.point_counter * self.sampling_period, self.sampling_period)
        polynom=np.poly1d(self.parameters)
        n = polynom(t)
        polynom_dot=np.polyder(polynom)
        n_dot = polynom_dot(t)
        return t, n, n_dot


    def __get_fun_exp(self):#coeefs [a,b,c,d]= a*e^(t/b+c)+d
        t = np.arange(0, self.point_counter * self.sampling_period, self.sampling_period) #same as sine
        a_p=self.parameters[0]
        b_p=self.parameters[1]
        c_p=self.parameters[2]
        d_p=self.parameters[3]
        n = a_p*np.exp(t/b_p+c_p)+d_p
        n_dot = a_p/b_p*np.exp(t/b_p+c_p)
        return t, n, n_dot