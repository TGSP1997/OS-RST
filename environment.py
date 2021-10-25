import numpy as np
import matplotlib.pyplot as plt

# utility and environment functions
def plot_sig(tt, signals, labels):
    plt.figure()
    for sig, lab in zip(signals, labels):
        plt.plot(tt, sig, label=lab)
    plt.legend()
    plt.xlabel('time', fontsize=20)
    plt.ylabel('value', fontsize=20)

def fwd_diff(tt, signal):
    diff = np.zeros(len(tt))
    for i in range(0, len(tt)-1):
        diff[i] = (signal[i+1] - signal[i]) / (tt[1] - tt[0])
    return diff