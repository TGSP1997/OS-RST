from scipy.signal import correlate
from enum import Enum

class Cost_Enum(Enum):
    MSE         = "Mean Squared Error"
    MINIMAX     = "Minimax"
    AVG_LOSS    = "average loss"
    PHASE_SHIFT = "phase shift"

class Cost:
    type        = Cost_Enum.MSE
    parameters  = []

    def __init__(self,type,parameters=[]):
        self.type       = type
        self.parameters = parameters

    # y = measured signals with noise
    # n = true signal
    def cost(self,y,n):
        match self.type:
            case Cost_Enum.MSE:
                return self.__cost_mse(y,n)
            case Cost_Enum.MINIMAX:
                return self.__cost_minimax(y,n)
            case Cost_Enum.AVG_LOSS:
                return self.__cost_avg_loss(y,n)
            case Cost_Enum.PHASE_SHIFT:
                return self.__cost_phase_shift(y,n)

####################################################################
######################## Cost Functions ############################
####################################################################

# Means squared error of a signal
    def __cost_mse(self,y, n):
        mse = 0
        for i in range(len(n)):
            mse = mse + (n[i] - y[i])**2
        mse = 1/len(n) * mse
        return mse

# verstehe ich entweder nicht tief genug oder ist in diesem Projekt nicht sinnvoll
    def __cost_minimax(self,y, n):
        return 0

# wie MSE nur mit Betrag?
    def __cost_avg_loss(self,y, n):
        avg_loss = 0
        for i in range(len(n)):
            avg_loss = avg_loss + abs(n[i] - y[i])
        avg_loss = 1/len(n) * avg_loss
        return avg_loss

# returns index of maximum correlation for phase shift
    def __cost_phase_shift(self,y, n):
        corr = correlate(y, n, mode='same')
        lag_index = corr.argmax()
        return lag_index