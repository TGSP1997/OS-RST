import numpy as np #https://www.youtube.com/watch?v=I_K7cVlg2Cc
import matplotlib.pyplot as plt
#from scipy.signal import savgol_filter
from matplotlib.widgets import Slider
from SL_adjusted_scipy_savgol_filter import *

current_v1=3
current_v2=99
current_v3=0
window=99

N = 99
T = 1.0 / 800.0



#generate signal with noise
x=np.linspace(0,2*np.pi,100)
y=np.sin(x)+np.cos(x)+np.random.random(100)

#y = 0.2*np.cos(2*np.pi*2*x) + np.cos(2*np.pi*20*x)

#filter
y_filtered=adjusted_savgol_filter(y,99,3)

#Plotting
fig=plt.figure()
ax=fig.subplots()
p=ax.plot(x,y,'b', label='Unfiltered')
p,=ax.plot(x,y_filtered,'g', label='Filtered')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.subplots_adjust(bottom=0.3)
#define the slider
ax_slide1 = plt.axes([0.25,0.15,0.65,0.03]) #xposition,y position, width,height
ax_slide2 = plt.axes([0.25,0.1,0.65,0.03])
ax_slide3 = plt.axes([0.25,0.05,0.65,0.03])
#properties of slider
win_len=Slider(ax_slide2,'Window Length',valmin=5,valmax=99,valinit=99,valstep=2)
pol_len=Slider(ax_slide1,'Polynom Grade',valmin=1,valmax=10,valinit=3,valstep=1)

pos_len=Slider(ax_slide3,'Position [Default: {}'.format(window//2)+']',valmin=0,valmax=99,valinit=window//2,valstep=1)
#creating updating function
def update(val2):
    current_v1=int(pol_len.val)
    current_v2=int(win_len.val)
    current_v3=int(pos_len.val)
    window=current_v2
    new_y=adjusted_savgol_filter(y,current_v2,current_v1,pos=current_v3)
    p.set_ydata(new_y)
    pos_len.label.set_text('Position [Default: {}'.format(window//2)+']')
    fig.canvas.draw() #redraw the figure
    print('pol:')
    print(current_v1)
    print('window:')
    print(current_v2)
    print('pos:')
    print(current_v3)

    new_yf2 = np.fft.fft(new_y)
    q.set_ydata(2.0/N * np.abs(new_yf2[0:N//2]))
    fig2.canvas.draw()
    

win_len.on_changed(update)
pol_len.on_changed(update)
pos_len.on_changed(update)


fig2=plt.figure(2)
ay=fig2.subplots()
yf = np.fft.fft(y)
yf2 = np.fft.fft(y_filtered)
xf = np.fft.fftfreq(N, T)[:N//2]
q=ay.plot(xf, 2.0/N * np.abs(yf[0:N//2]),'b')
q,=ay.plot(xf, 2.0/N * np.abs(yf2[0:N//2]),'g')
plt.grid()


plt.show()

