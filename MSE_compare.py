from matplotlib.pyplot import show
from scipy import signal
from numpy import ma

from input_function import *
from noises import *
from filter import *
from plot_sig import *
from cost import *

def plot_sig_mse(signals, labels, title, ref, ymax):
    labels = ['0.1', '0.2', '0.3', '0.4', '0.5']

    x = np.arange(len(labels))  # the label locations
    width = 0.16  # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(x - 2*width, signals[0]/ref*100, width, label='Wiener', color=(0, 0.467, 0.682, 1))
    ax.bar(x - width, signals[1]/ref*100, width, label='Kalman', color=(0.969, 0.663, 0.255, 1))
    ax.bar(x  , signals[2]/ref*100, width, label='Savgol', color=(0.58, 0.765, 0.337, 1))
    ax.bar(x + width , signals[3]/ref*100, width, label='Butter', color=(0.518, 0.812, 0.929, 1))
    ax.bar(x+ 2*width  , signals[4]/ref*100, width, label='Exp.', color=(0.584, 0.106, 0.506, 1))
    # Add some text for labels, title and custom x-axis tick labels, etc.

    rects = ax.patches
    # Make some labels.
    
    box_labels = ["","","","","","","","","","","","","","","","","","","","","","","","",""]
    i = 0
    for sigs in signals:        
        for sig, r in zip(sigs, ref):
            box_labels[i] = "%.1e" % sig
            i = i+1
    for rect, label in zip(rects, box_labels):
        height = rect.get_height()
        #ax.text(
         #   rect.get_x() + rect.get_width() / 2, 0.025*ymax, label, ha="center", va="bottom", rotation = "vertical", color=(1,1,1,1), fontweight = "semibold", fontsize=8
        #)
    
    ax.set_xlabel(r'noise $\sigma$', fontsize=16)
    ax.set_ylabel('Relatives MSE [%]', fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x, labels, fontsize =15)
    plt.yticks( fontsize =15)
    plt.ylim(0, ymax)
    ax.legend()

    fig.tight_layout()

def plot_sig_mse_absolut(signals, labels, title, ref, ymax):
    labels = ['0.1', '0.2', '0.3', '0.4', '0.5']
    x = np.arange(len(labels))  # the label locations
    width = 0.16  # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(x - 2*width, signals[0], width, label='Wiener', color=(0, 0.467, 0.682, 1))
    ax.bar(x - width, signals[1], width, label='Kalman', color=(0.969, 0.663, 0.255, 1))
    ax.bar(x  , signals[2], width, label='Savgol', color=(0.58, 0.765, 0.337, 1))
    ax.bar(x + width , signals[3], width, label='Butter', color=(0.518, 0.812, 0.929, 1))
    ax.bar(x+ 2*width  , signals[4], width, label='Exp.', color=(0.584, 0.106, 0.506, 1))
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(r'noise $\sigma$', fontsize=16)
    ax.set_ylabel('MSE', fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x, labels, fontsize =14)
    plt.yticks( fontsize =14)
    plt.ylim(0, ymax)
    ax.legend()
    fig.tight_layout()

data_csv = np.genfromtxt(r'filter_compare.csv', delimiter=',')#r'C:\Users\Sandro\Documents\Code\OS\OS-RST\filter_compare.csv'
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

MSE_sine_white_butter = data_csv[39:44, 2]
MSE_sine_brown_butter = data_csv[39:44, 3]
MSE_sine_quant_butter = data_csv[39:44, 4]
MSE_poly_white_butter = data_csv[39:44, 5]
MSE_poly_brown_butter = data_csv[39:44, 6]
MSE_poly_quant_butter = data_csv[39:44, 7]

MSE_sine_white_exp = data_csv[44:49, 2]
MSE_sine_brown_exp = data_csv[44:49, 3]
MSE_sine_quant_exp = data_csv[44:49, 4]
MSE_poly_white_exp = data_csv[44:49, 5]
MSE_poly_brown_exp = data_csv[44:49, 6]
MSE_poly_quant_exp = data_csv[44:49, 7]

MSE_sine_white_ref = data_csv[63:68, 2]
MSE_sine_brown_ref = data_csv[63:68, 3]
MSE_sine_quant_ref = data_csv[63:68, 4]
MSE_poly_white_ref = data_csv[63:68, 5]
MSE_poly_brown_ref = data_csv[63:68, 6]
MSE_poly_quant_ref = data_csv[63:68, 7]

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

MSE_sine_white_butter_dot = data_csv[51:56, 2]
MSE_sine_brown_butter_dot = data_csv[51:56, 3]
MSE_sine_quant_butter_dot = data_csv[51:56, 4]
MSE_poly_white_butter_dot = data_csv[51:56, 5]
MSE_poly_brown_butter_dot = data_csv[51:56, 6]
MSE_poly_quant_butter_dot = data_csv[51:56, 7]

MSE_sine_white_exp_dot = data_csv[56:61, 2]
MSE_sine_brown_exp_dot = data_csv[56:61, 3]
MSE_sine_quant_exp_dot = data_csv[56:61, 4]
MSE_poly_white_exp_dot = data_csv[56:61, 5]
MSE_poly_brown_exp_dot = data_csv[56:61, 6]
MSE_poly_quant_exp_dot = data_csv[56:61, 7]

MSE_sine_white_ref_dot = data_csv[69:74, 2]
MSE_sine_brown_ref_dot = data_csv[69:74, 3]
MSE_sine_quant_ref_dot = data_csv[69:74, 4]
MSE_poly_white_ref_dot = data_csv[69:74, 5]
MSE_poly_brown_ref_dot = data_csv[69:74, 6]
MSE_poly_quant_ref_dot = data_csv[69:74, 7]

############### NICHT abgeleiteter Sinus
# white sine
plot_sig_mse([MSE_sine_white_wiener, MSE_sine_white_kalman,MSE_sine_white_savgol,MSE_sine_white_butter,MSE_sine_white_exp], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Sinus, Weißes R., Filter', MSE_sine_white_ref, 55)
# brown sine
plot_sig_mse([MSE_sine_brown_wiener, MSE_sine_brown_kalman,MSE_sine_brown_savgol,MSE_sine_brown_butter,MSE_sine_brown_exp], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Sinus, Braunes R., Filter', MSE_sine_brown_ref, 110)
# quant sine
plot_sig_mse([MSE_sine_quant_wiener, MSE_sine_quant_kalman,MSE_sine_quant_savgol,MSE_sine_quant_butter,MSE_sine_quant_exp], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Sinus, Quantisierungsr., Filter', MSE_sine_quant_ref, 150)

############### NICHT abgeleitetes Polynom
# white poly
plot_sig_mse([MSE_poly_white_wiener, MSE_poly_white_kalman,MSE_poly_white_savgol,MSE_poly_white_butter,MSE_poly_white_exp], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Polynom, Weißes R., Filter', MSE_poly_white_ref, 20)
# brown poly
plot_sig_mse([MSE_poly_brown_wiener, MSE_poly_brown_kalman,MSE_poly_brown_savgol,MSE_poly_brown_butter,MSE_poly_brown_exp], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Polynom, Braunes R., Filter', MSE_poly_brown_ref, 30)
# quant poly
plot_sig_mse([MSE_poly_quant_wiener, MSE_poly_quant_kalman,MSE_poly_quant_savgol,MSE_poly_quant_butter,MSE_poly_quant_exp], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Polynom, Quantisierungsr., Filter', MSE_poly_quant_ref, 120)

############### abgeleiteter Sinus
# white sine
plot_sig_mse([MSE_sine_white_wiener_dot, MSE_sine_white_kalman_dot,MSE_sine_white_savgol_dot,MSE_sine_white_butter_dot,MSE_sine_white_exp_dot], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Sinus, Weißes R., Filter & Ableitung', MSE_sine_white_ref_dot, 0.2)
# brown sine
plot_sig_mse([MSE_sine_brown_wiener_dot, MSE_sine_brown_kalman_dot, MSE_sine_brown_savgol_dot, MSE_sine_brown_butter_dot, MSE_sine_brown_exp_dot,], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Sinus, Braunes R., Filter & Ableitung', MSE_sine_brown_ref_dot, 10)
# quant sine
plot_sig_mse([MSE_sine_quant_wiener_dot, MSE_sine_quant_kalman_dot, MSE_sine_quant_savgol_dot, MSE_sine_quant_butter_dot, MSE_sine_quant_exp_dot,], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Sinus, Quantisierungsr., Filter & Ableitung', MSE_sine_quant_ref_dot, 5)

############### abgeleitetes Polynom
# white poly
plot_sig_mse([MSE_poly_white_wiener_dot, MSE_poly_white_kalman_dot, MSE_poly_white_savgol_dot, MSE_poly_white_butter_dot, MSE_poly_white_exp_dot], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Polynom, Weißes R., Filter & Ableitung', MSE_poly_white_ref_dot, 0.01)
# brown poly
plot_sig_mse([MSE_poly_brown_wiener_dot, MSE_poly_brown_kalman_dot, MSE_poly_brown_savgol_dot, MSE_poly_brown_butter_dot, MSE_poly_brown_exp_dot], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Polynom, Braunes R., Filter & Ableitung', MSE_poly_brown_ref_dot, 0.15)
# quant poly
plot_sig_mse([MSE_poly_quant_wiener_dot, MSE_poly_quant_kalman_dot, MSE_poly_quant_savgol_dot, MSE_poly_quant_butter_dot, MSE_poly_quant_exp_dot], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Polynom, Quantisierungsr., Filter & Ableitung', MSE_poly_quant_ref_dot, 0.3)

from os import path
outpath = r"Bilder_Vergleich/Prozentual/"
for i in plt.get_fignums():
        plt.figure(i).savefig(path.join(outpath,"figure_{0}.png".format(i)))

plt.show()

############### NICHT abgeleiteter Sinus
# white sine
plot_sig_mse_absolut([MSE_sine_white_wiener, MSE_sine_white_kalman,MSE_sine_white_savgol,MSE_sine_white_butter,MSE_sine_white_exp], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Sinus, Weißes R., Filter', MSE_sine_white_ref, 0.05)
# brown sine
plot_sig_mse_absolut([MSE_sine_brown_wiener, MSE_sine_brown_kalman,MSE_sine_brown_savgol,MSE_sine_brown_butter,MSE_sine_brown_exp], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Sinus, Braunes R., Filter', MSE_sine_brown_ref, 0.25)
# quant sineSinus
plot_sig_mse_absolut([MSE_sine_quant_wiener, MSE_sine_quant_kalman,MSE_sine_quant_savgol,MSE_sine_quant_butter,MSE_sine_quant_exp], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Sinus, Quantisierungsr., Filter', MSE_sine_quant_ref, 0.02)

############### NICHT abgeleitetes Polynom
# white poly
plot_sig_mse_absolut([MSE_poly_white_wiener, MSE_poly_white_kalman,MSE_poly_white_savgol,MSE_poly_white_butter,MSE_poly_white_exp], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Polynom, Weißes R., Filter', MSE_poly_white_ref, 0.075)
# brown poly
plot_sig_mse_absolut([MSE_poly_brown_wiener, MSE_poly_brown_kalman,MSE_poly_brown_savgol,MSE_poly_brown_butter,MSE_poly_brown_exp], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Polynom, Braunes R., Filter', MSE_poly_brown_ref, 0.15)
# quant poly
plot_sig_mse_absolut([MSE_poly_quant_wiener, MSE_poly_quant_kalman,MSE_poly_quant_savgol,MSE_poly_quant_butter,MSE_poly_quant_exp], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Polynom, Quantisierungsr., Filter', MSE_poly_quant_ref, 0.025)

############### abgeleiteter Sinus
# white sine
plot_sig_mse_absolut([MSE_sine_white_wiener_dot, MSE_sine_white_kalman_dot,MSE_sine_white_savgol_dot,MSE_sine_white_butter_dot,MSE_sine_white_exp_dot], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Sinus, Weißes R. , Filter & Ableitung', MSE_sine_white_ref_dot, 60)
# brown sine
plot_sig_mse_absolut([MSE_sine_brown_wiener_dot, MSE_sine_brown_kalman_dot, MSE_sine_brown_savgol_dot, MSE_sine_brown_butter_dot, MSE_sine_brown_exp_dot,], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Sinus, Braunes R., Filter & Ableitung', MSE_sine_brown_ref_dot, 100)
# quant sine
plot_sig_mse_absolut([MSE_sine_quant_wiener_dot, MSE_sine_quant_kalman_dot, MSE_sine_quant_savgol_dot, MSE_sine_quant_butter_dot, MSE_sine_quant_exp_dot,], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Sinus, Quantisierungsr., Filter & Ableitung', MSE_sine_quant_ref_dot, 40)

############### abgeleitetes Polynom
# white poly
plot_sig_mse_absolut([MSE_poly_white_wiener_dot, MSE_poly_white_kalman_dot, MSE_poly_white_savgol_dot, MSE_poly_white_butter_dot, MSE_poly_white_exp_dot], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Polynom, Weißes R., Filter & Ableitung', MSE_poly_white_ref_dot, 4)
# brown poly
plot_sig_mse_absolut([MSE_poly_brown_wiener_dot, MSE_poly_brown_kalman_dot, MSE_poly_brown_savgol_dot, MSE_poly_brown_butter_dot, MSE_poly_brown_exp_dot], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Polynom, Braunes R., Filter & Ableitung', MSE_poly_brown_ref_dot, 4)
# quant poly
plot_sig_mse_absolut([MSE_poly_quant_wiener_dot, MSE_poly_quant_kalman_dot, MSE_poly_quant_savgol_dot, MSE_poly_quant_butter_dot, MSE_poly_quant_exp_dot], ['Wiener', 'Kalman','Savgol','Butter','Exp'], 'Polynom, Quantisierungsr., Filter & Ableitung', MSE_poly_quant_ref_dot, 4)

from os import path
outpath = r"Bilder_Vergleich/Absolut/"
for i in plt.get_fignums():
        plt.figure(i).savefig(path.join(outpath,"figure_{0}.png".format(i)))

plt.show()