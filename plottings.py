import matplotlib.pyplot as plt

# plot of multiple time series in one graph
def plot_time_sig(tt, signals, labels):
    plt.figure()
    for sig, lab in zip(signals, labels):
        plt.plot(tt, sig, label=lab)
    plt.legend()
    plt.xlabel('time', fontsize=20)
    plt.ylabel('value', fontsize=20)