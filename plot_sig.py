import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.core.fromnumeric import size
from scipy.ndimage.measurements import label
from filter import *
from matplotlib.widgets import Slider
from enum import Enum
import numpy as np


class Plot_Enum(Enum):
    MULTI = "multi"
    SLIDER = 'Slider'
    SUBPLOT = "subplot"

class Plot_Sig:
    type        = Plot_Enum.MULTI
    title       = ""
    parameters  = []

    def __init__(self,type, title, parameters):
        self.type               = type
        self.title              = title
        self.parameters         = parameters
        self.fig                = 0
       
        

    def plot_sig(self,t,signals, labels, time_domain=True):
        match self.type:
            case Plot_Enum.MULTI:
                self.__plot_sig_multi(t,signals, labels, time_domain)
            case Plot_Enum.SUBPLOT:
                self.__plot_sig_subplot(t,signals,labels)
    
    def plot_slider(self,t,signals, labels,parameters,filter): #parameters as array, differrent for ech filter 
        match self.type:
            case Plot_Enum.SLIDER:
                self.parameters=parameters
                self.__plot_sig_slider(t,signals, labels,filter)

####################################################################
######################## Plot Functions ############################
####################################################################

    def __plot_sig_multi(self,t,signals,labels, time_domain):
        plt.figure()
        plt.suptitle(self.title)
        for sig, lab in zip(signals, labels):
            plt.plot(t, sig, label=lab)
        plt.legend()
        if time_domain == True:
            plt.xlabel('time', fontsize=20)
        else:
            plt.xlabel('normalized frequency', fontsize=20)
        plt.ylabel('value', fontsize=20)
        plt.title(self.title)
      
    def __plot_sig_subplot(self,t,signals,labels):
        fig = plt.figure(constrained_layout=True)
        spec = gridspec.GridSpec(size(signals,0),1, figure = fig, wspace = 0)
        
        for i in range(size(signals,0)):
            ax = fig.add_subplot(spec[i,0])
            ax.plot(t, signals[i], label=labels[i])
            plt.legend()
            plt.ylabel('value', fontsize=20)
            plt.xlim(min(t),max(t))
            plt.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top = False,
                labelbottom = False
            )
        plt.xlabel('time', fontsize=20)
        plt.tick_params(
                axis="x",
                which="both",
                bottom=True,
                top = False,
                labelbottom = True
            )

    def __plot_sig_slider(self,t,signals,labels,filter): #self.parameters[0]=m self.parameters[1]=polyorder
        self.fig=plt.figure()
        fig_plots=self.fig.subplots()
        fig_plots.set_xlabel('time')
        fig_plots.set_ylabel('value')
        match filter.type:
            case Filter_Enum.SAVGOL: 
                window_length=2*self.parameters[0]+1
                p=fig_plots.plot(t,signals[0],'b', label=labels[0])
                p,=fig_plots.plot(t,signals[1],'g', label=labels[1])
                plt.subplots_adjust(bottom=0.3)
                fig_plots_slide1 = plt.axes([0.25,0.15,0.65,0.03]) #xposition,y position, width,height
                fig_plots_slide2 = plt.axes([0.25,0.1,0.65,0.03])
                
                if (window_length%2)==0:
                    raise ValueError('Window_Length must be a odd number')
                slider_1=Slider(fig_plots_slide1,'Polynom Grade',valmin=1,valmax=self.parameters[1]+10,valinit=self.parameters[1],valstep=1)
                slider_2=Slider(fig_plots_slide2,'Window Length',valmin=1,valmax=2*self.parameters[0]+1+50,valinit=2*self.parameters[0]+1,valstep=2)
               
                def __update_SAVGOL(val):
                    current_v1=int(slider_1.val)
                    current_v2=int(slider_2.val)
                    half,_ = divmod(current_v2, 2)
                    filtered_signal=filter.filter_fun(t,signals[0],para=[half,current_v1]) 
                    p.set_ydata(filtered_signal)
                    self.fig.canvas.draw() #redraw the figure

                slider_1.on_changed(__update_SAVGOL)
                slider_2.on_changed(__update_SAVGOL)
                fig_plots.legend()
                plt.show()
                
            

            case Filter_Enum.PT1:        
                p=fig_plots.plot(t,signals[0],'b', label=labels[0])
                p,=fig_plots.plot(t,signals[1],'g', label=labels[1])
                plt.subplots_adjust(bottom=0.3)
                fig_plots_slide1 = plt.axes([0.25,0.15,0.65,0.03]) #xposition,y position, width,height
                slider_1=Slider(fig_plots_slide1,'Cutoff Frequency',valmin=1,valmax=1000,valinit=self.parameters[0],valstep=10)

                def __update_PT1(val):
                    current_v1=np.array([slider_1.val])
                    filtered_signal=filter.filter_fun(t, signals[0], para = current_v1)
                    p.set_ydata(filtered_signal)
                    self.fig.canvas.draw() #redraw the figure
                    
                slider_1.on_changed(__update_PT1)
                fig_plots.legend()
                plt.show()
                
            case Filter_Enum.DIFF_QUOTIENT:
                p=fig_plots.plot(t,signals[0],'b', label=labels[0])
                p,=fig_plots.plot(t,signals[1],'g', label=labels[1])
                plt.subplots_adjust(bottom=0.3)
                fig_plots_slide1 = plt.axes([0.25,0.15,0.65,0.03]) #xposition,y position, width,height
                slider_1=Slider(fig_plots_slide1,'Difference',valmin=0,valmax=self.parameters[0]+100,valinit=self.parameters[0],valstep=1)

                def __update_DIFF_QUOTIENT(val):
                    current_v1=np.array([slider_1.val])
                    filtered_signal=filter.filter_fun(t, signals[0], para = current_v1)
                    p.set_ydata(filtered_signal)
                    self.fig.canvas.draw() #redraw the figure
                    
                slider_1.on_changed(__update_DIFF_QUOTIENT)
                fig_plots.legend()
                plt.show()
              
            case Filter_Enum.BROWN_HOLT:
                p=fig_plots.plot(t,signals[0],'b', label=labels[0])
                p,=fig_plots.plot(t,signals[1],'g', label=labels[1])
                plt.subplots_adjust(bottom=0.3)
                fig_plots_slide1 = plt.axes([0.25,0.15,0.65,0.03]) #xposition,y position, width,height
                slider_1=Slider(fig_plots_slide1,'Alpha',valmin=0,valmax=1,valinit=self.parameters[0],valstep=0.005)

                def __update_BROWN_HOLT(val):
                    current_v1=np.array([slider_1.val])
                    filtered_signal=filter.filter_fun(t, signals[0], para = current_v1)
                    p.set_ydata(filtered_signal)
                    self.fig.canvas.draw() #redraw the figure
                    
                slider_1.on_changed(__update_BROWN_HOLT)
                fig_plots.legend()
                plt.show()

            #case Filter_Enum.BUTTERWORTH:
                
            #case Filter_Enum.CHEBYCHEV:
               
            #case Filter_Enum.ROB_EX_DIFF:
                
            case Filter_Enum.KALMAN:
                p=fig_plots.plot(t,signals[0],'b', label=labels[0])
                p,=fig_plots.plot(t,signals[1],'g', label=labels[1])
                plt.subplots_adjust(bottom=0.3)
                fig_plots_slide1 = plt.axes([0.25,0.15,0.65,0.03]) #xposition,y position, width,height
                slider_1=Slider(fig_plots_slide1,'Kalman Process Noise',valmin=0,valmax=10**8,valinit=int(self.parameters[2]),valstep=0.005*(10**8))

                def __update_KALMAN(val):
                    current_v1=np.array([slider_1.val])
                    filtered_signal,_,_=filter.filter_fun(t, signals[0], para = [self.parameters[0],self.parameters[1],current_v1,self.parameters[3]])
                    p.set_ydata(filtered_signal)
                    self.fig.canvas.draw() #redraw the figure
                    
                slider_1.on_changed(__update_KALMAN)
                fig_plots.legend()
                plt.show()
     
    
####################################################################
######################## Nicht angefasst ###########################
####################################################################




#x=np.linspace(0,2*np.pi,100)
#y=np.sin(x)+np.cos(x)+np.random.random(100)
#y_filtered,_,_=savgol_smooth(y,99,3,160,deriv=0)
#sig=[y, y_filtered]
#plot_slider(filter.SAVGOL,x,sig) #savgol_smooth(y,99,3,160,deriv=0)
