from cost import Cost_Enum
from scipy.optimize import minimize
import scipy.fftpack
from input_function import *
from noises import *
from filter import *
from plot_sig import *
from cost import *

noise_std_dev   = 0.1
noise_std_dev_2   = 0.3
noise_std_dev_3   = 0.6
alpha           = 0.4


sine    = Input_Function(Input_Enum.SINE, [1,0.4, 0, 0],point_counter=1e3)
polynom = Input_Function(Input_Enum.POLYNOM, [1,2,3,4]) #coeefs in descending order 2x^2+1 = [2,0,1]
exp     = Input_Function(Input_Enum.EXP, [1,2,0,0]) #coeefs [a,b,c,d]= a*e^(t/b+c)+d
white   = Noise(Noise_Enum.WHITE, noise_std_dev)
white_2   = Noise(Noise_Enum.WHITE, noise_std_dev_2)
white_3   = Noise(Noise_Enum.WHITE, noise_std_dev_3)
plot    = Plot_Sig(Plot_Enum.MULTI, "Overview", [])
plot_sub= Plot_Sig(Plot_Enum.SUBPLOT, "Overview", [])
cost    = Cost(Cost_Enum.MSE)
plot_s  = Plot_Sig(Plot_Enum.SLIDER, "Detailed View with Slider",[])#only one slider window can be open at a time
savgol  = Filter(Filter_Enum.SAVGOL, parameters=None) #para=[m,polorder,diff=number]

time, true_sine, true_sine_dot = sine.get_fun()
norm_freq = time[:round(len(time)/2)] / (time[-1] - time[0])
y = white.apply_noise(true_sine)
y_2 = white_2.apply_noise(true_sine)
y_3 = white_3.apply_noise(true_sine)

def filter_cost(para_in, t, y, true_y, filter, cost):
        y_hat_pt1 = filter.filter_fun(t, y, para = para_in)
        return cost.cost(y_hat_pt1, true_y)


def sagol_filter_cost(para_in,t,y,true_y,filter,cost):
        print(para_in)
        para_in=para_in.astype(int)
        y_hat_savgol=filter.filter_fun(t,y,para=para_in)
        new_y_hat_savgol=[x for x in y_hat_savgol if x is not None]

        return cost.cost(new_y_hat_savgol,true_y[(len(true_y)-len(new_y_hat_savgol)):])


savgol_filter_para=[2,2]


def own_minimize(t,y,true_y,filter,cost):
        minimum_cost_z=None
        minimum_cost= None
        para_out=[]
        for i in range(1,12):
                for j in range(0,200):
                        
                        if i>=(2*j+1):
                                continue
                        y_hat_savgol=filter.filter_fun(t,y,para=[j,i])
                        new_y_hat_savgol=[x for x in y_hat_savgol if x is not None]

                        minimum_cost_z=cost.cost(new_y_hat_savgol,true_y[(len(true_y)-len(new_y_hat_savgol)):])
                        #print('var',i,2*j+1,minimum_cost_z)
                        if minimum_cost==None:
                                minimum_cost=minimum_cost_z
                        if minimum_cost_z<=minimum_cost:
                                minimum_cost=minimum_cost_z
                                para_out=['p:',i,'m//2:',j,'window:',2*j+1,minimum_cost]
        return para_out






print(own_minimize(time, y, true_sine, savgol ,cost))
print(own_minimize(time, y_2, true_sine, savgol ,cost))
print(own_minimize(time, y_3, true_sine, savgol ,cost))
savgol_filter_para_s=[10,3]
y_hat_savgol_s=savgol.filter_fun(time,y,para=savgol_filter_para_s)
plot_s.plot_slider(time,[y, y_hat_savgol_s],['noisy sine','savgol smoothed'],savgol_filter_para_s,savgol)
y_hat_savgol_s_2=savgol.filter_fun(time,y_2,para=savgol_filter_para_s)
plot_s.plot_slider(time,[y_2, y_hat_savgol_s_2],['noisy sine_2','savgol smoothed'],savgol_filter_para_s,savgol)
y_hat_savgol_s_3=savgol.filter_fun(time,y_3,para=savgol_filter_para_s)
plot_s.plot_slider(time,[y_3, y_hat_savgol_s_3],['noisy sine_3','savgol smoothed'],savgol_filter_para_s,savgol)
#################################

new_y_hat_savgol=[x for x in y_hat_savgol_s if x is not None]
new_time=time[(len(time)-len(new_y_hat_savgol)):]
#ps=np.abs(np.fft.fftshift(np.fft.fft(new_time)))**2
#ps_1 = np.abs(np.fft.fftshift(np.fft.fft(new_y_hat_savgol)))**2
#fpix = np.arange(ps.shape[0]) - ps.shape[0]//2
#plot.plot_sig(fpix, [ps_1], ["ps"])

#f = np.fft.rfftfreq(4*savgol_filter_para_s[0]+1, d=1/1e-3)
#Y=2*np.abs(np.fft.rfft(new_y_hat_savgol, 4*savgol_filter_para_s[0]+1))/(1e-3*2)
#plot.plot_sig(f, [Y], ["ps"])

#sampling_freq=160
#sampling_interval=1/sampling_freq
#f_trans=np.fft.fft(new_y_hat_savgol)/len(new_y_hat_savgol)
#f_trans= f_trans[range(int(len(new_y_hat_savgol)/2))] # Exclude sampling frequency  

#tpCount     = len(new_y_hat_savgol)
#values      = np.arange(int(tpCount/2))
#timePeriod  = tpCount/sampling_freq
#frequencies = values/timePeriod
#plot.plot_sig(frequencies, [abs(f_trans)], ["ps"])
N=600
T = 1.0 / 800.0
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0//(2.0*T), N//2)
#plot.plot_sig(xf,[ 2.0/N * np.abs(yf[:N//2])], ["ps"])

N=1000
tstep=1e-3
t=np.linspace(0,(N-1)*tstep,N)
fstep=1/(tstep*N)
f=np.linspace(0,(N-1)*fstep,N-2*savgol_filter_para_s[0])
y=new_y_hat_savgol
X=np.fft.fft(y)
X_mag=np.abs(X)/N
plot.plot_sig(f,[X_mag], ["ps"])
########################################

plt.show()

