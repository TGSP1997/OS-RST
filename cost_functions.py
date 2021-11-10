# Means squared error of a signal
def mean_squared_error(meas_sig, true_sig):
    mse = 0
    for i in range(len(true_sig)):
        mse = mse + (true_sig[i] - meas_sig[i])**2
    mse = 1/len(true_sig) * mse
    return mse