import random
import torch
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

## Metrics
class MSE():
    def __init__(self):
        self.name = "mse"
    def __call__(self, y, yhat):
        mse = np.square(np.array(yhat) - np.array(y)).mean()
        return mse

class corr():
    def __init__(self):
        self.name = "correlation"
    
    def __call__(self, y, yhat):
        if len(y.shape) > 1:
            corrs = []
            for i in range(yhat.shape[1]):
                corrs.append(np.corrcoef(yhat[:, i], y[:, i])[0, 1])
            
            # return(np.mean(np.array(corrs)))
            return corrs
        else:
            return np.corrcoef(yhat, y)[0, 1]
        
class R2():
    def __init__(self):
        self.name = "R2"

    def __call__(self, y, yhat):
        if len(y.shape) > 1:
            corrs = []
            for i in range(yhat.shape[1]):
                corrs.append(r2_score(yhat[:, i], y[:, i]))
            
            # return(np.mean(np.array(corrs)))
            return corrs
        else:
            return r2_score(yhat, y)
        
        
def plot_predictions(y, yhat, title, labels, legend = True, perf = True, metric = corr()):
    plt.figure(figsize = [15, 10])
    nplots = yhat.shape[1]
    plt.suptitle(title)
    if nplots == 2:
        sp1 = 1
        sp2 = 2
    elif nplots ==4 or nplots == 5:
        sp1 = 2
        sp2 = 2
    for i in range(nplots):
        plt.subplot(sp1, sp2, i+1) 
       
        if np.ptp(yhat[:, i]) > np.ptp(y[:, i]):
            plt.plot(yhat[:, i], color = 'r', label = "Predicted")
            plt.plot(y[:, i], color = 'b', label = "Measured" )
        else:
            plt.plot(y[:, i], color = 'b', label = "Measured" )
            plt.plot(yhat[:, i], color = 'r', label = "Predicted")

        if legend:
            plt.legend(loc= "upper left")
        plt.title(labels[i])
        Ylim = getYLim([y[:, i], yhat[:, i]]) 
        location = (2, Ylim[0])
        plt.ylim([Ylim[0]*1.1, Ylim[1]*1.1])
        if perf:
            plt.annotate(f"{metric.name}: {metric(y[:, i],yhat[:, i])}", location)


def getYLim(sets):
    ## Calculate the y limits for plotting
    Min = float('inf')
    Max = 0
    for i in sets:
        Min = min(min(i), Min)
        Max = max(max(i), Max)
    
    
    if np.isnan(Max) or np.isinf(Max):
        Max = 100
        
    if np.isnan(Min) or np.isinf(Min):
        Min = 0
    
    buffer = .1 * (Max - Min)
    return np.array([Min - buffer, Max + buffer])
    
    
def set_seed(seed):
    # Set Random Seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups 