import random
import torch
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import dataset
import models

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
        
        
def plot_predictions(y, yhat, title, legend = True, perf = True, metric = corr()):
    labels = ['First Finger Position', 'Second Finger Position', 'First Finger Velocity', 'Second Finger Velocity']
    nplots = yhat.shape[1]
    if nplots == 5: 
        nplots = 4
    
    plt.figure(figsize = [15, 10])
    plt.suptitle(title)
    
    if nplots == 2:
        sp1 = 1
        sp2 = 2
    elif nplots ==4:
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
            
        plt.xlabel("Timepoint")
        if i < 2:
            plt.ylabel("Position")
        elif i >= 2:
            plt.ylabel("Velocity")
            


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
        
def get_lbda(X, Y, intercept, lbda_start = 0, lbda_end = 50, lbda_step = 5, metric = corr()):
    
    X_train, Y_train, X_val, Y_val = dataset.get_val_set(X, Y)
    
    lbdas = np.arange(lbda_start, lbda_end + lbda_step, lbda_step)
    perfs = np.empty([len(lbdas), Y_train.shape[1]])
    for i, lbda in enumerate(lbdas): 
        model = models.RidgeRegression(lbda, intercept)
        model.train(X_train, Y_train)
        yhat = model.run(X_val, None)
        perfs[i, :] = metric(Y_val, yhat)
        
    plt.figure()
    plt.suptitle("Ridge regression performance vs Lambda value")
    
    nplots = perfs.shape[1]
    if nplots == 2:
        sp1 = 1
        sp2 = 2
    elif nplots == 4:
        sp1 = 2
        sp2 = 2
        

    for i in range(nplots):
        plt.subplot(sp1, sp2, i+1) 
       
        plt.plot(perfs[:, i], color = 'black')
        plt.xlabel("Lambda")
        plt.ylabel(f"{metric.name}")
        plt.xticks(np.arange(len(lbdas)), lbdas)  # Custom labels
    
    plt.tight_layout()
    
    return lbdas[np.argmax(perfs, axis = 0)[0]]


        