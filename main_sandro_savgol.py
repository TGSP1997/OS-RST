from matplotlib.pyplot import show
from scipy.optimize import minimize
from scipy.fft import fft, ifft, fftfreq

from input_function import *
from noises import *
from filter import *
from plot_sig import *
from cost import *

###########################################################
#Functions
def own_minimize(t,y,true_y,filter,cost,diff=0):
        minimum_cost_z=None
        minimum_cost= None
        para_out=[]
        for i in range(1,20):
                for j in range(0,100):
                        
                        if i>=(2*j+1):
                                continue
                        y_hat_savgol=filter.filter_fun(t,y,para=[j,i,diff])
                        new_y_hat_savgol=[x for x in y_hat_savgol if x is not None]

                        minimum_cost_z=cost.cost(new_y_hat_savgol,true_y[(len(true_y)-len(new_y_hat_savgol)):])
                        #print('var',i,2*j+1,minimum_cost_z)
                        if minimum_cost==None:
                                minimum_cost=minimum_cost_z
                        if minimum_cost_z<=minimum_cost:
                                minimum_cost=minimum_cost_z
                                para_out=[j,i,diff]
        return para_out,minimum_cost
###########################################################
#0. Vorbereitung
step_size       = 2e-3#5e-3
#step_size       = 1e-3
noise_std_dev   = 0.5
plot1  = Plot_Sig(Plot_Enum.FILTER1, "Savitzky Golay | Harmonic Signal",[])

plot2  = Plot_Sig(Plot_Enum.FILTER2, "Savitzky Golay | Derivative Harmonic Signal",[])

###########################################################
#1. Filtereigenschaften Sinus
sine    = Input_Function(Input_Enum.SINE, [1, 0.5, 0, 0], sampling_period = step_size, point_counter=500)#200
#sine    = Input_Function(Input_Enum.SINE, [1, 0.4, 0, 0], sampling_period = step_size, point_counter=1000)
savgol  = Filter(Filter_Enum.SAVGOL, parameters=None) #para=[m,polorder,diff=number]
white   = Noise(Noise_Enum.WHITE, noise_std_dev)
brown   = Noise(Noise_Enum.BROWN, noise_std_dev)
quant   = Noise(Noise_Enum.QUANT, noise_std_dev)
cost    = Cost(Cost_Enum.MSE)

time, x, x_dot = sine.get_fun()
y_white = white.apply_noise(x)
y_brown = brown.apply_noise(x)
y_quant = quant.apply_noise(x)

savgol_para_white,cost_white = own_minimize(time, y_white, x, savgol ,cost)
x_hat_min_white=savgol.filter_fun(time,y_white,para=savgol_para_white)
standard_cost_white = cost.cost(y_white,x)
print('1.','white',savgol_para_white,'cost',cost_white)

savgol_para_brown,cost_brown=own_minimize(time, y_brown, x, savgol ,cost)
x_hat_min_brown=savgol.filter_fun(time,y_brown,para=savgol_para_brown)
standard_cost_brown = cost.cost(y_brown,x)
print('1.','brown',savgol_para_brown,'cost',cost_brown)

savgol_para_quant,cost_quant=own_minimize(time, y_quant, x, savgol ,cost)
x_hat_min_quant=savgol.filter_fun(time,y_quant,para=savgol_para_quant)
standard_cost_quant = cost.cost(y_quant,x)
print('1.','quant',savgol_para_quant,'cost',cost_quant)

box_label_white = '\n'.join((
        r'White Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'Polyorder=%i Window=%i' % (savgol_para_white[1],savgol_para_white[0]*2+1 ),
        r'$MSE_{Filter}=%.2e$' % (cost_white, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_white, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_white/standard_cost_white, )))

box_label_brown = '\n'.join((
        r'Brown Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'Polyorder=%i Window=%i' % (savgol_para_brown[1],savgol_para_brown[0]*2+1 ),
        r'$MSE_{Filter}=%.2e$' % (cost_brown, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_brown, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_brown/standard_cost_brown, )))

box_label_quant = '\n'.join((
        r'Quantisation Noise',
        r'$stepsize=%.2f$' % (noise_std_dev, ),
        r'Polyorder=%i Window=%i' % (savgol_para_quant[1],savgol_para_quant[0]*2+1 ),
        r'$MSE_{Filter}=%.2e$' % (cost_quant, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_quant, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_quant/standard_cost_quant, )))

plot1.plot_sig(time,[x,y_white,y_brown,y_quant,x_hat_min_white,x_hat_min_brown,x_hat_min_quant],[r'$f(t) = \mathrm{sin}\left(2\pi\cdot\frac{t}{0.5\,\mathrm{s}}\right)$',
'Signal with White Noise',
'Signal with Brown Noise',
'Signal with Quantisation Noise',
'Savgol Smoothing',
'Savgol Smoothing',
'Savgol Smoothing',
box_label_white,box_label_brown,box_label_quant],True)

###########################################################
#2. Ableitungseigenschaften Sinus

savgol_para_white,cost_white = own_minimize(time, y_white, x_dot, savgol ,cost,diff=1)
#savgol_para_white=np.append(savgol_para_white,1)
x_hat_min_white=savgol.filter_fun(time,y_white,para=savgol_para_white)
standard_cost_white = cost.cost(y_white,x_dot)
print('2.','white',savgol_para_white,'cost',cost_white)

savgol_para_brown,cost_brown=own_minimize(time, y_brown, x_dot, savgol ,cost,diff=1)
x_hat_min_brown=savgol.filter_fun(time,y_brown,para=savgol_para_brown)
standard_cost_brown = cost.cost(y_brown,x_dot)
print('2.','brown',savgol_para_brown,'cost',cost_brown)

savgol_para_quant,cost_quant=own_minimize(time, y_quant, x_dot, savgol ,cost,diff=1)
x_hat_min_quant=savgol.filter_fun(time,y_quant,para=savgol_para_quant)
standard_cost_quant = cost.cost(y_quant,x_dot)
print('2.','quant',savgol_para_quant,'cost',cost_quant)

box_label_white = '\n'.join((
        r'White Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'Polyorder=%i Window=%i' % (savgol_para_white[1],savgol_para_white[0]*2+1 ),
        r'$MSE_{Filter}=%.2e$' % (cost_white, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_white, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_white/standard_cost_white, )))

box_label_brown = '\n'.join((
        r'Brown Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'Polyorder=%i Window=%i' % (savgol_para_brown[1],savgol_para_brown[0]*2+1 ),
        r'$MSE_{Filter}=%.2e$' % (cost_brown, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_brown, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_brown/standard_cost_brown, )))

box_label_quant = '\n'.join((
        r'Quantisation Noise',
        r'$stepsize=%.2f$' % (noise_std_dev, ),
        r'Polyorder=%i Window=%i' % (savgol_para_quant[1],savgol_para_quant[0]*2+1 ),
        r'$MSE_{Filter}=%.2e$' % (cost_quant, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_quant, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_quant/standard_cost_quant, )))
y_white_dot = np.diff(y_white, append = 0)/step_size
y_brown_dot = np.diff(y_brown, append = 0)/step_size
y_quant_dot = np.diff(y_quant, append = 0)/step_size
plot2.plot_sig(time,[x_dot,y_white_dot,y_brown_dot,y_quant_dot,x_hat_min_white,x_hat_min_brown,x_hat_min_quant],[r'$\frac{df}{dt}(t) = \left(\frac{2\pi}{0.5 \mathrm{s}}\right)\mathrm{cos}\left(2\pi\cdot\frac{t}{0.5 \mathrm{s}}\right)$',
'Diff of signal with White Noise',
'Diff of signal with Brown Noise',
'Diff of signal with Quantisation Noise',
'Savgol Smoothing and Differentation',
'Savgol Smoothing and Differentation',
'Savgol Smoothing and Differentation',
box_label_white,box_label_brown,box_label_quant],True)
###########################################################
#3. Filtereigenschaften Polynom
polynome = Input_Function(Input_Enum.POLYNOM, [4, -6, 3, 0]) #[100,-150,50,0]
plot1  = Plot_Sig(Plot_Enum.FILTER3, "Savitzky Golay | Polynomial Signal",[])
plot2  = Plot_Sig(Plot_Enum.FILTER4, "Savitzky Golay | Derivative Polynomial Signal",[])
time, x, x_dot = polynome.get_fun()
y_white = white.apply_noise(x)
y_brown = brown.apply_noise(x)
y_quant = quant.apply_noise(x)


savgol_para_white,cost_white = own_minimize(time, y_white, x, savgol ,cost)
x_hat_min_white=savgol.filter_fun(time,y_white,para=savgol_para_white)
standard_cost_white = cost.cost(y_white,x)
print('3.','white',savgol_para_white,'cost',cost_white)

savgol_para_brown,cost_brown=own_minimize(time, y_brown, x, savgol ,cost)
x_hat_min_brown=savgol.filter_fun(time,y_brown,para=savgol_para_brown)
standard_cost_brown = cost.cost(y_brown,x)
print('3.','brown',savgol_para_brown,'cost',cost_brown)

savgol_para_quant,cost_quant=own_minimize(time, y_quant, x, savgol ,cost)
x_hat_min_quant=savgol.filter_fun(time,y_quant,para=savgol_para_quant)
standard_cost_quant = cost.cost(y_quant,x)
print('3.','quant',savgol_para_quant,'cost',cost_quant)

box_label_white = '\n'.join((
        r'White Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'Polyorder=%i Window=%i' % (savgol_para_white[1],savgol_para_white[0]*2+1 ),
        r'$MSE_{Filter}=%.2e$' % (cost_white, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_white, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_white/standard_cost_white, )))

box_label_brown = '\n'.join((
        r'Brown Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'Polyorder=%i Window=%i' % (savgol_para_brown[1],savgol_para_brown[0]*2+1 ),
        r'$MSE_{Filter}=%.2e$' % (cost_brown, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_brown, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_brown/standard_cost_brown, )))

box_label_quant = '\n'.join((
        r'Quantisation Noise',
        r'$stepsize=%.2f$' % (noise_std_dev, ),
        r'Polyorder=%i Window=%i' % (savgol_para_quant[1],savgol_para_quant[0]*2+1 ),
        r'$MSE_{Filter}=%.2e$' % (cost_quant, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_quant, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_quant/standard_cost_quant, )))

plot1.plot_sig(time,[x,y_white,y_brown,y_quant,x_hat_min_white,x_hat_min_brown,x_hat_min_quant],[r'$f(t) = \frac{4}{\mathrm{s}^3}\cdot t^3 - \frac{6}{\mathrm{s}^2}\cdot t^2 + \frac{3}{\mathrm{s}}\cdot t + 0$',
'Signal with White Noise',
'Signal with Brown Noise',
'Signal with Quantisation Noise',
'Savgol Smoothing',
'Savgol Smoothing',
'Savgol Smoothing',
box_label_white,box_label_brown,box_label_quant],True)
###########################################################
#4. Ableitungseigenschaften Polynom

savgol_para_white,cost_white = own_minimize(time, y_white, x_dot, savgol ,cost,diff=1)
x_hat_min_white=savgol.filter_fun(time,y_white,para=savgol_para_white)
standard_cost_white = cost.cost(y_white,x_dot)
print('4.','white',savgol_para_white,'cost',cost_white)

savgol_para_brown,cost_brown=own_minimize(time, y_brown, x_dot, savgol ,cost,diff=1)
x_hat_min_brown=savgol.filter_fun(time,y_brown,para=savgol_para_brown)
standard_cost_brown = cost.cost(y_brown,x_dot)
print('4.','brown',savgol_para_brown,'cost',cost_brown)

savgol_para_quant,cost_quant=own_minimize(time, y_quant, x_dot, savgol ,cost,diff=1)
x_hat_min_quant=savgol.filter_fun(time,y_quant,para=savgol_para_quant)
standard_cost_quant = cost.cost(y_quant,x_dot)
print('4.','quant',savgol_para_quant,'cost',cost_quant)

box_label_white = '\n'.join((
        r'White Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'Polyorder=%i Window=%i' % (savgol_para_white[1],savgol_para_white[0]*2+1 ),
        r'$MSE_{Filter}=%.2e$' % (cost_white, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_white, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_white/standard_cost_white, )))

box_label_brown = '\n'.join((
        r'Brown Noise',
        r'$\sigma_{Noise}=%.2f$' % (noise_std_dev, ),
        r'Polyorder=%i Window=%i' % (savgol_para_brown[1],savgol_para_brown[0]*2+1 ),
        r'$MSE_{Filter}=%.2e$' % (cost_brown, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_brown, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_brown/standard_cost_brown, )))

box_label_quant = '\n'.join((
        r'Quantisation Noise',
        r'$stepsize=%.2f$' % (noise_std_dev, ),
        r'Polyorder=%i Window=%i' % (savgol_para_quant[1],savgol_para_quant[0]*2+1 ),
        r'$MSE_{Filter}=%.2e$' % (cost_quant, ),
        r'$MSE_{Noise}=%.2e$' % (standard_cost_quant, ),
        r'$r_{MSE}=%.2f$ %%' % (100*cost_quant/standard_cost_quant, )))

y_white_dot = np.diff(y_white, append = 0)/step_size
y_brown_dot = np.diff(y_brown, append = 0)/step_size
y_quant_dot = np.diff(y_quant, append = 0)/step_size
plot2.plot_sig(time,[x_dot,y_white_dot,y_brown_dot,y_quant_dot,x_hat_min_white,x_hat_min_brown,x_hat_min_quant],[r'$\frac{df}{dt}(t) = \frac{12}{\mathrm{s}^3}\cdot t^2 - \frac{12}{\mathrm{s}^2}\cdot t + \frac{3}{\mathrm{s}}$',
'Diff of signal with White Noise',
'Diff of signal with Brown Noise',
'Diff of signal with Quantisation Noise',
'Savgol Smoothing and Differentation',
'Savgol Smoothing and Differentation',
'Savgol Smoothing and Differentation',
box_label_white,box_label_brown,box_label_quant],True)
###########################################################
#5.

###########################################################
#6. Impuls Filtereigenschaften - Eigene Implementation
point_counter   = 200
u = np.zeros(int(point_counter))
u[10] = 1
t = np.linspace(0,1,num = int(point_counter))
m=10
y1      = savgol.filter_fun(t,u,para = [ m, 1 ])
y3      = savgol.filter_fun(t,u,para = [ m, 3 ])
y5      = savgol.filter_fun(t,u,para = [ m, 5 ])
y7      = savgol.filter_fun(t,u,para = [ m, 7 ])
y9      = savgol.filter_fun(t,u,para = [ m, 9 ])
y10     = savgol.filter_fun(t,u,para = [ m, 10 ])#####


y1=[i if i is not None else 0 for i in y1]
y3=[i if i is not None else 0 for i in y3]
y5=[i if i is not None else 0 for i in y5]
y7=[i if i is not None else 0 for i in y7]
y9=[i if i is not None else 0 for i in y9]
y10=[i if i is not None else 0 for i in y10]

plot_bode = Plot_Sig(Plot_Enum.BODE,"Bode Plot",parameters = 0)
plot_bode.plot_sig(t,[[u,u,u,u,u,u],[y1,y3,y5,y7,y9,y10]],[
        "Window=21 p=1", 
        "Window=21 p=3",
        "Window=21 p=5",
        "Window=21 p=7",
        "Window=21 p=9",
        "Window=21 p=10",])

###########################################################
#7. Impuls Filtereigenschaften - Scipy Implementation
point_counter=200
u = np.zeros(int(point_counter))
u[10] = 1
t = np.linspace(0,1,num = int(point_counter))
from scipy.signal import savgol_filter
y1      = savgol_filter(u,21, 1 )
y3      = savgol_filter(u, 21, 3 )
y5      = savgol_filter(u, 21, 5 )
y7      = savgol_filter(u, 21, 7 )
y9      = savgol_filter(u, 21, 9 )
y10     = savgol_filter(u, 21, 10)

plot_bode = Plot_Sig(Plot_Enum.BODE,"Bode Plot",parameters = 0)
plot_bode.plot_sig(t,[[u,u,u,u,u,u],[y1,y3,y5,y7,y9,y10]],[
        "Scipy Window=21 p=1", 
        "Scipy Window=21 p=3",
        "Scipy Window=21 p=5",
        "Scipy Window=21 p=7",
        "Scipy Window=21 p=9",
        "Scipy Window=21 p=10",])

###########################################################

#plot_s  = Plot_Sig(Plot_Enum.SLIDER, "Detailed View with Slider",[])
#plot_s.plot_slider(time,[y_white, x_hat_min_white],['noisy sine','savgol smoothed'],[20,3],savgol)
plt.show()