import numpy as np
import sympy
from enum import Enum

# generate sine input function and its derivative
def sine_input(sampling_period, point_count, sine_period):
    time = np.arange(0, point_count * sampling_period, sampling_period)
    sine = np.sin(2*np.pi/sine_period * time)
    diff_sine = np.cos(2*np.pi/sine_period * time)

    return [sine, diff_sine]

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