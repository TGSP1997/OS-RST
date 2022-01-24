from matplotlib.pyplot import show
from scipy import signal
from numpy import ma

from input_function import *
from noises import *
from filter import *
from plot_sig import *
from cost import *

def plot_sig_mse(signals, labels, title):
    labels = ['0.1', '0.2', '0.3', '0.4', '0.5']
    x = np.arange(len(labels))  # the label locations
    width = 0.16  # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(x - 2*width, signals[0], width, label='Wiener')
    ax.bar(x - width, signals[1], width, label='Kalman')
    ax.bar(x  , signals[2], width, label='Savgol')
    ax.bar(x + width , signals[2], width, label='jonas1')
    ax.bar(x+ 2*width  , signals[2], width, label='jonas2')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(r'noise $\sigma$')
    ax.set_ylabel('MSE')
    ax.set_title(title)
    ax.set_xticks(x, labels)
    ax.legend()
    fig.tight_layout()

data_csv = np.genfromtxt(r'C:\Users\Sandro\Documents\Code\OS\OS-RST\filter_compare.csv', delimiter=',')#r'C:\Users\Sandro\Documents\Code\OS\OS-RST\filter_compare.csv'
std_devs = data_csv[1:6, 1]
# NICHT abgeleitete MSEs
MSE_sine_white_wiener = data_csv[1:6, 2]
MSE_sine_brown_wiener = data_csv[1:6, 3]
MSE_sine_quant_wiener = data_csv[1:6, 4]
MSE_poly_white_wiener = data_csv[1:6, 5]
MSE_poly_brown_wiener = data_csv[1:6, 6]
MSE_poly_quant_wiener = data_csv[1:6, 7]
MSE_sine_white_kalman = data_csv[6:11, 2]
MSE_sine_brown_kalman = data_csv[6:11, 3]
MSE_sine_quant_kalman = data_csv[6:11, 4]
MSE_poly_white_kalman = data_csv[6:11, 5]
MSE_poly_brown_kalman = data_csv[6:11, 6]
MSE_poly_quant_kalman = data_csv[6:11, 7]

MSE_sine_white_savgol = data_csv[25:30, 2]
MSE_sine_brown_savgol = data_csv[25:30, 3]
MSE_sine_quant_savgol = data_csv[25:30, 4]
MSE_poly_white_savgol = data_csv[25:30, 5]
MSE_poly_brown_savgol = data_csv[25:30, 6]
MSE_poly_quant_savgol = data_csv[25:30, 7]


# abgeleitete MSEs
MSE_sine_white_wiener_dot = data_csv[13:18, 2]
MSE_sine_brown_wiener_dot = data_csv[13:18, 3]
MSE_sine_quant_wiener_dot = data_csv[13:18, 4]
MSE_poly_white_wiener_dot = data_csv[13:18, 5]
MSE_poly_brown_wiener_dot = data_csv[13:18, 6]
MSE_poly_quant_wiener_dot = data_csv[13:18, 7]
MSE_sine_white_kalman_dot = data_csv[18:23, 2]
MSE_sine_brown_kalman_dot = data_csv[18:23, 3]
MSE_sine_quant_kalman_dot = data_csv[18:23, 4]
MSE_poly_white_kalman_dot = data_csv[18:23, 5]
MSE_poly_brown_kalman_dot = data_csv[18:23, 6]
MSE_poly_quant_kalman_dot = data_csv[18:23, 7]

MSE_sine_white_savgol_dot = data_csv[32:37, 2]
MSE_sine_brown_savgol_dot = data_csv[32:37, 3]
MSE_sine_quant_savgol_dot = data_csv[32:37, 4]
MSE_poly_white_savgol_dot = data_csv[32:37, 5]
MSE_poly_brown_savgol_dot = data_csv[32:37, 6]
MSE_poly_quant_savgol_dot = data_csv[32:37, 7]


############### NICHT abgeleiteter Sinus
# white sine
plot_sig_mse([MSE_sine_white_wiener, MSE_sine_white_kalman,MSE_sine_white_savgol], ['Wiener', 'Kalman','Savgol'], 'MSEs of white noise at sine')
# brown sine
plot_sig_mse([MSE_sine_brown_wiener, MSE_sine_brown_kalman,MSE_sine_brown_savgol], ['Wiener', 'Kalman','Savgol'], 'MSEs of brown noise at sine')
# quant sine
plot_sig_mse([MSE_sine_quant_wiener, MSE_sine_quant_kalman,MSE_sine_quant_savgol], ['Wiener', 'Kalman','Savgol'], 'MSEs of quant noise at sine')

############### NICHT abgeleitetes Polynom
# white poly
plot_sig_mse([MSE_poly_white_wiener, MSE_poly_white_kalman,MSE_poly_white_savgol], ['Wiener', 'Kalman','Savgol'], 'MSEs of white noise at poly')
# brown poly
plot_sig_mse([MSE_poly_brown_wiener, MSE_poly_brown_kalman,MSE_poly_brown_savgol], ['Wiener', 'Kalman','Savgol'], 'MSEs of brown noise at poly')
# quant poly
plot_sig_mse([MSE_poly_quant_wiener, MSE_poly_quant_kalman,MSE_poly_quant_savgol], ['Wiener', 'Kalman','Savgol'], 'MSEs of quant noise at poly')

############### abgeleiteter Sinus
# white sine
plot_sig_mse([MSE_sine_white_wiener_dot, MSE_sine_white_kalman_dot,MSE_sine_white_savgol_dot], ['Wiener', 'Kalman','Savgol'], 'MSEs of white noise at sine_dot')
# brown sine
plot_sig_mse([MSE_sine_brown_wiener_dot, MSE_sine_brown_kalman_dot, MSE_sine_brown_savgol_dot], ['Wiener', 'Kalman','Savgol'], 'MSEs of brown noise at sine_dot')
# quant sine
plot_sig_mse([MSE_sine_quant_wiener_dot, MSE_sine_quant_kalman_dot, MSE_sine_quant_savgol_dot], ['Wiener', 'Kalman','Savgol'], 'MSEs of quant noise at sine_dot')

############### abgeleitetes Polynom
# white poly
plot_sig_mse([MSE_poly_white_wiener_dot, MSE_poly_white_kalman_dot, MSE_poly_white_savgol_dot], ['Wiener', 'Kalman','Savgol'], 'MSEs of white noise at poly_dot')
# brown poly
plot_sig_mse([MSE_poly_brown_wiener_dot, MSE_poly_brown_kalman_dot, MSE_poly_brown_savgol_dot], ['Wiener', 'Kalman','Savgol'], 'MSEs of brown noise at poly_dot')
# quant poly
plot_sig_mse([MSE_poly_quant_wiener_dot, MSE_poly_quant_kalman_dot, MSE_poly_quant_savgol_dot], ['Wiener', 'Kalman','Savgol'], 'MSEs of quant noise at poly_dot')

plt.show()