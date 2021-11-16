import matplotlib.pyplot as plt
from filter import *
from matplotlib.widgets import Slider


# plot of multiple time series in one graph
def plot_time_sig(tt, signals, labels):
    plt.figure()
    for sig, lab in zip(signals, labels):
        plt.plot(tt, sig, label=lab)
    plt.legend()
    plt.xlabel('time', fontsize=20)
    plt.ylabel('value', fontsize=20)

    

def plot_slider(enum,tt,signals): #match case wouldnt work signals[unfiltered,filtered]
    fig=plt.figure()
    fig_plots=fig.subplots()
    fig_plots.set_xlabel('Time')
    fig_plots.set_ylabel('Value')

    if enum==filter.SAVGOL:
        p=fig_plots.plot(tt,signals[0],'b', label='Unfiltered')
        p,=fig_plots.plot(tt,signals[1],'g', label='Savgol Filtered')
        plt.subplots_adjust(bottom=0.3)
        fig_plots_slide1 = plt.axes([0.25,0.15,0.65,0.03]) #xposition,y position, width,height
        fig_plots_slide2 = plt.axes([0.25,0.1,0.65,0.03])
        fig_plots_slide3 = plt.axes([0.25,0.05,0.65,0.03])
        pol_len=Slider(fig_plots_slide1,'Polynom Grade',valmin=1,valmax=10,valinit=3,valstep=1)
        win_len=Slider(fig_plots_slide2,'Window Length',valmin=5,valmax=99,valinit=99,valstep=2)
        deriv_len=Slider(fig_plots_slide3,'Derivative',valmin=0,valmax=5,valinit=0,valstep=1)

        def update_SAVGOL(val):
            current_v1=int(pol_len.val)
            current_v2=int(win_len.val)
            current_v3=int(deriv_len.val)
            filtered_signal,_,_=savgol_smooth(signals[0],current_v2,current_v1,160,deriv=current_v3) 
            p.set_ydata(filtered_signal)
            fig.canvas.draw() #redraw the figure
            print('pol:')
            print(current_v1)
            print('window:')
            print(current_v2)
            print('deriv:')
            print(current_v3)
        
        pol_len.on_changed(update_SAVGOL)
        win_len.on_changed(update_SAVGOL)
        deriv_len.on_changed(update_SAVGOL)
        
    else:
         raise ValueError('Function does not exist')
    fig_plots.legend()
    plt.show()
    '''
    elif enum==filter.WIENER:
        def update_WIENER(val):
    elif enum==filter.DIFF_QUOTIENT:
        def update_DIFF_QUOTIENT(val):
    elif enum==filter.BROWN_HOLT:
        def update_BROWN_HOLT(val):
    elif enum==filter.BUTTERWORTH:
        def update_BUTTERWORTH(val):
    elif enum==filter.CHEBYCHEV:
        def update_CHEBYCHEV(val):
    elif enum==filter.ROB_EX_DIFF:
        def update_ROB_EX_DIFF(val):
    '''
    
    



#x=np.linspace(0,2*np.pi,100)
#y=np.sin(x)+np.cos(x)+np.random.random(100)
#y_filtered,_,_=savgol_smooth(y,99,3,160,deriv=0)
#sig=[y, y_filtered]
#plot_slider(filter.SAVGOL,x,sig) #savgol_smooth(y,99,3,160,deriv=0)
