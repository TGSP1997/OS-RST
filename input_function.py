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


    # Returns t, y, y_dot
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
        t = np.arange(0, self.point_counter * self.sampling_period, self.sampling_period)
        y = self.parameters[0]*np.sin(2*np.pi/self.parameters[1] * t + self.parameters[2]) + self.parameters[3]
        y_dot = np.cos(2*np.pi/self.parameters[1] * t)
        return t, y, y_dot

        
    def __get_fun_polynom(self):
        t = 0
        y = 0
        y_dot = 0
        return t, y, y_dot

    def __get_fun_exp(self):
        t = 0
        y = 0
        y_dot = 0
        return t, y, y_dot


    

'''
class inputs(Enum):
    sine='sine'
    polynom='polynom'
    exp='exp'

def sine_input(parameters,point_count,sampling_period): #parameters [a,f,o]
    time = np.arange(0, point_count * sampling_period, sampling_period)
    sine = parameters[0]*np.sin(2*np.pi/parameters[1] * time)+parameters[2]
    diff_sine = np.cos(2*np.pi/parameters[1] * time)
    return [time,sine,diff_sine]

def polynom_input(parameters,point_count,sampling_period): #parameters [a^n,a^n-1,..,a^0]
    time = np.arange(0, point_count * sampling_period, sampling_period)
    polynom=np.poly1d(parameters)
    diff_polynom=np.polyder(polynom)
    return [time, polynom, diff_polynom]


#die ableitung macht noch probleme, bitte drueberschauen
def exp_input(parameters,point_count,sampling_period): #parmeters [a,t,o]
    time = np.arange(0, point_count * sampling_period, sampling_period)
    exp  = parameters[0]*sympy.exp(time/parameters[1])+parameters[2]
    diff_exp=exp.diff(time, 1)

    return [time,exp,diff_exp]
    

def get_inputs(enum,parameters,point_count,sampling_period): #sympy needed,see imports
    if enum==inputs.sine.value:
        input=sine_input(parameters,point_count,sampling_period)
    elif enum==inputs.polynom.value: 
        input=polynom_input(parameters,point_count,sampling_period)
    elif enum==inputs.exp.value:
        input=exp_input(parameters,point_count,sampling_period)
    else:
         raise ValueError('Function does not exist')

    print(input) #for testing
    return input

get_inputs('sine',[3,2,1],3,50)
get_inputs('polynom',[3,2,1],3,50)
get_inputs('exp',[3,2,1],3,50)
'''