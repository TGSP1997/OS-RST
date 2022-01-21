from matplotlib.pyplot import show
from scipy import signal
from numpy import ma

from input_function import *
from noises import *
from filter import *
from plot_sig import *
from cost import *

data_csv = np.genfromtxt('filter_compare.csv', delimiter=',')
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


labels = ['0.1', '0.2', '0.3', '0.4', '0.5']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

############### NICHT abgeleiteter Sinus
# white sine
fig_white_sine, ax_white_sine = plt.subplots()
rects1 = ax_white_sine.bar(x - width/2, MSE_sine_white_wiener, width, label='Wiener')
rects2 = ax_white_sine.bar(x + width/2, MSE_sine_white_kalman, width, label='Kalman')
# Add some text for labels, title and custom x-ax_white_sineis tick labels, etc.
ax_white_sine.set_xlabel(r'noise $\sigma$')
ax_white_sine.set_ylabel('MSE')
ax_white_sine.set_title('MSEs of white noise at sine')
ax_white_sine.set_xticks(x, labels)
ax_white_sine.legend()
fig_white_sine.tight_layout()

# brown sine
fig_brown_sine, ax_brown_sine = plt.subplots()
rects1 = ax_brown_sine.bar(x - width/2, MSE_sine_brown_wiener, width, label='Wiener')
rects2 = ax_brown_sine.bar(x + width/2, MSE_sine_brown_kalman, width, label='Kalman')
# Add some text for labels, title and custom x-ax_brown_sineis tick labels, etc.
ax_brown_sine.set_xlabel(r'noise $\sigma$')
ax_brown_sine.set_ylabel('MSE')
ax_brown_sine.set_title('MSEs of brown noise at sine')
ax_brown_sine.set_xticks(x, labels)
ax_brown_sine.legend()
fig_brown_sine.tight_layout()

# quant sine
fig_quant_sine, ax_quant_sine = plt.subplots()
rects1 = ax_quant_sine.bar(x - width/2, MSE_sine_quant_wiener, width, label='Wiener')
rects2 = ax_quant_sine.bar(x + width/2, MSE_sine_quant_kalman, width, label='Kalman')
# Add some text for labels, title and custom x-ax_quant_sineis tick labels, etc.
ax_quant_sine.set_xlabel(r'noise $\sigma$')
ax_quant_sine.set_ylabel('MSE')
ax_quant_sine.set_title('MSEs of quant noise at sine')
ax_quant_sine.set_xticks(x, labels)
ax_quant_sine.legend()
fig_quant_sine.tight_layout()

############### NICHT abgeleitetes Polynom
# white poly
fig_white_poly, ax_white_poly = plt.subplots()
rects1 = ax_white_poly.bar(x - width/2, MSE_poly_white_wiener, width, label='Wiener')
rects2 = ax_white_poly.bar(x + width/2, MSE_poly_white_kalman, width, label='Kalman')
# Add some text for labels, title and custom x-ax_white_polyis tick labels, etc.
ax_white_poly.set_xlabel(r'noise $\sigma$')
ax_white_poly.set_ylabel('MSE')
ax_white_poly.set_title('MSEs of white noise at poly')
ax_white_poly.set_xticks(x, labels)
ax_white_poly.legend()
fig_white_poly.tight_layout()

# brown poly
fig_brown_poly, ax_brown_poly = plt.subplots()
rects1 = ax_brown_poly.bar(x - width/2, MSE_poly_brown_wiener, width, label='Wiener')
rects2 = ax_brown_poly.bar(x + width/2, MSE_poly_brown_kalman, width, label='Kalman')
# Add some text for labels, title and custom x-ax_brown_polyis tick labels, etc.
ax_brown_poly.set_xlabel(r'noise $\sigma$')
ax_brown_poly.set_ylabel('MSE')
ax_brown_poly.set_title('MSEs of brown noise at poly')
ax_brown_poly.set_xticks(x, labels)
ax_brown_poly.legend()
fig_brown_poly.tight_layout()

# quant poly
fig_quant_poly, ax_quant_poly = plt.subplots()
rects1 = ax_quant_poly.bar(x - width/2, MSE_poly_quant_wiener, width, label='Wiener')
rects2 = ax_quant_poly.bar(x + width/2, MSE_poly_quant_kalman, width, label='Kalman')
# Add some text for labels, title and custom x-ax_quant_polyis tick labels, etc.
ax_quant_poly.set_xlabel(r'noise $\sigma$')
ax_quant_poly.set_ylabel('MSE')
ax_quant_poly.set_title('MSEs of quant noise at poly')
ax_quant_poly.set_xticks(x, labels)
ax_quant_poly.legend()
fig_quant_poly.tight_layout()

plt.show()