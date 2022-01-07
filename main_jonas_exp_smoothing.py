from re import S
from matplotlib.pyplot import show
from matplotlib import ticker, cm
from scipy.optimize import minimize
from scipy.fft import fft, ifft, fftfreq
from scipy import signal
from numpy import ma

from input_function import *
from noises import *
from filter import *
from plot_sig import *
from cost import *

noise_std_dev   = 0.1
alpha           = 0.4
freq            = 10 / (1000 / 2) # Normalisierte Grenzfrequenz mit w = fc / (fs / 2)

sine    = Input_Function(Input_Enum.SINE, [1, 0.2, 0, 0])
brown1   = Filter(Filter_Enum.BROWN_HOLT, [alpha,1])
brown2   = Filter(Filter_Enum.BROWN_HOLT, [alpha,2])
brown3   = Filter(Filter_Enum.BROWN_HOLT, [alpha,3])
cost    = Cost(Cost_Enum.MSE)
plot    = Plot_Sig(Plot_Enum.SUBPLOT,"Butter",0)
plot_s  = Plot_Sig(Plot_Enum.SLIDER, "Detailed View with Slider",[])#only one slider window can be open at a time
white   = Noise(Noise_Enum.WHITE, noise_std_dev)


def filter_cost(para_in, t, y, x, filter, cost):
        y_hat = filter.filter_fun(t, y, para = [para_in, filter.parameters[1]])
        return cost.cost(y_hat, x)
def filter_cost_diff(para_in, t, y, x_dot, filter, cost):
        y_hat_dot = filter.filter_diff(t, y, para = [filter.parameters[0], filter.parameters[1], para_in])
        return cost.cost(y_hat_dot, x_dot)

t, x, x_dot = sine.get_fun()
y = white.apply_noise(x)

a_min1 = minimize(filter_cost,alpha,args=(t, y, x, brown1 ,cost))
y_hat_min1 = brown1.filter_fun(t,y,para = [a_min1.x,brown1.parameters[1]])

a_min2 = minimize(filter_cost,alpha,args=(t, y, x, brown2 ,cost))
y_hat_min2 = brown2.filter_fun(t,y,para = [a_min2.x,brown2.parameters[1]])

a_min3 = minimize(filter_cost,alpha,args=(t, y, x, brown3 ,cost))
y_hat_min3 = brown3.filter_fun(t,y,para = [a_min3.x,brown3.parameters[1]])


standard_cost = cost.cost(y,x)
exp_cost1 = cost.cost(y_hat_min1,x)
exp_cost2 = cost.cost(y_hat_min2,x)
exp_cost3 = cost.cost(y_hat_min3,x)

plot.plot_sig(t,[x,y,y_hat_min1,y_hat_min2,y_hat_min3],['sine',
'noisy sine (' + str(standard_cost) + ')',
'Exp. Sm. (' + str(exp_cost1) + ')',
'Exp. Sm. (' + str(exp_cost2) + ')',
'Exp. Sm. (' + str(exp_cost3) + ')'],True)
plt.show()

# Optimierte Ableitung

y_dot = np.diff(y, prepend=0)

b_min = minimize(filter_cost_diff,alpha,args=(t, y, x_dot, brown1 ,cost))
y_dot_hat_min1 = brown1.filter_diff(t,y,para = [a_min1.x,brown1.parameters[1],b_min.x])

b_min = minimize(filter_cost_diff,alpha,args=(t, y, x_dot, brown2 ,cost))
y_dot_hat_min2 = brown2.filter_diff(t,y,para = [a_min2.x,brown2.parameters[1],b_min.x])

b_min = minimize(filter_cost_diff,alpha,args=(t, y, x_dot, brown3 ,cost))
y_dot_hat_min3 = brown3.filter_diff(t,y,para = [a_min3.x,brown3.parameters[1],b_min.x])

exp_cost1 = cost.cost(y_dot_hat_min1,x_dot)
exp_cost2 = cost.cost(y_dot_hat_min2,x_dot)
exp_cost3 = cost.cost(y_dot_hat_min3,x_dot)

plot.plot_sig(t,[x_dot,y_dot,y_dot_hat_min1,y_dot_hat_min2,y_dot_hat_min3],['sine',
'noisy sine (' + str(standard_cost) + ')',
'Exp. Sm. (' + str(exp_cost1) + ')',
'Exp. Sm. (' + str(exp_cost2) + ')',
'Exp. Sm. (' + str(exp_cost3) + ')'],True)
plt.show()

# >Variation der Alpha-Werte bei der Ableitung
# Robustheitsbetrachtung

alpha = np.linspace(0,1,51)
beta = np.linspace(0,1,51)

res1 = np.zeros([len(alpha),len(beta)])
res2 = np.zeros([len(alpha),len(beta)])
res3 = np.zeros([len(alpha),len(beta)])

for i in range(len(alpha)):
        for j in range(len(beta)):
                y_dot_hat = brown1.filter_diff(t,y,para = [alpha[i], brown1.parameters[1], beta[j]])
                res1[i][j] = cost.cost(y_dot_hat,x_dot)
                y_dot_hat = brown2.filter_diff(t,y,para = [alpha[i], brown2.parameters[1], beta[j]])
                res2[i][j] = cost.cost(y_dot_hat,x_dot)
                y_dot_hat = brown3.filter_diff(t,y,para = [alpha[i], brown3.parameters[1], beta[j]])
                res3[i][j] = cost.cost(y_dot_hat,x_dot)

plot.plot_sig(alpha,[res1[:,20],res2[:,20], res3[:,20]],['First Order', 'Second Order', 'Third Order'],True)

res1 = ma.masked_where(res1 > 0.01, res1)
res2 = ma.masked_where(res2 > 0.01, res2)
res3 = ma.masked_where(res3 > 0.01, res3)

lev_exp = np.linspace(-5,-2,31)
levs = np.power(10, lev_exp)

fig1 = plt.figure()
cs = plt.contourf(alpha, beta, res1, levs)
cbar = fig1.colorbar(cs)
fig1.suptitle('First Order', fontsize=20)

fig2 = plt.figure()
cs = plt.contourf(alpha, beta, res2, levs)
cbar = fig2.colorbar(cs)
fig2.suptitle('Second Order', fontsize=20)

fig3 = plt.figure()
cs = plt.contourf(alpha, beta, res3, levs)
cbar = fig3.colorbar(cs)
fig3.suptitle('Third Order', fontsize=20)

plt.show()