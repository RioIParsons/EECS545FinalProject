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
        
        
def plot_predictions(y, yhat, title, perf = True, metric = corr(), color = 'r', end = -1):
    labels = ['First Finger Position', 'Second Finger Position', 'First Finger Velocity', 'Second Finger Velocity']
    nplots = yhat.shape[1]
    if nplots == 5: 
        nplots = 4
    
    plt.figure(figsize = [10, 8])
    plt.suptitle(title)
    
    if nplots == 2:
        sp1 = 1
        sp2 = 2
    elif nplots ==4:
        sp1 = 2
        sp2 = 2
        

    for i in range(nplots):
        plt.subplot(sp1, sp2, i+1) 
       
        plt.plot(yhat[:end, i], color = color, label = "Predicted")
        plt.plot(y[:end, i], color = 'black', label = "Measured" )
       
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
    plt.tight_layout()

            


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
        
def get_lbda(X, Y, intercept, lbda_start = 0, lbda_end = 5, lbda_step = 5, metric = corr()):
    
    X_train, Y_train, X_val, Y_val = dataset.get_val_set(X, Y)
    
    lbdas = np.arange(lbda_start, lbda_end + lbda_step, lbda_step)
    perfs = np.empty([len(lbdas), Y_train.shape[1]])
    for i, lbda in enumerate(lbdas): 
        model = models.RidgeRegression(lbda, intercept)
        model.train_model(X_train, Y_train)
        yhat = model.run_model(X_val, None)
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
    best_lbdas = lbdas[np.argmax(perfs, axis = 0)]
    lbda = np.mean(best_lbdas)
    
    return lbda


def plot_train_val_loss(train_losses,  val_losses):
    plt.figure()
    plt.plot(train_losses, label = "Train Losses")
    plt.plot(val_losses, label = "Val Losses")
    plt.legend()
    plt.ylabel("MSE loss")
    plt.xlabel("Epochs")
    
def compare_perfs(Y, yhats, model_labels, colors, start = 0, end = -1):
    labels = ['First Finger Position', 'Second Finger Position', 'First Finger Velocity', 'Second Finger Velocity']
    nplots = Y.shape[1]
    if nplots == 5: 
        nplots = 4
    
    plt.figure(figsize = [15, 10])
    plt.suptitle("Comparison of Model Performances")
    
    if nplots == 2:
        sp1 = 1
        sp2 = 2
    elif nplots ==4:
        sp1 = 2
        sp2 = 2
        

    for i in range(nplots):
        plt.subplot(sp1, sp2, i+1) 
       
        for j in range(len(yhats)):
            plt.plot(yhats[j][start:end, i], color = colors[j], label = model_labels[j] )
            
        plt.plot(Y[start:end, i], color = 'black', label = "Ground Truth")

        plt.title(labels[i])
        data = [d[:, i] for d in yhats]
        data.append(Y[:, :i])

        plt.xlabel("Timepoint")
        if i < 2:
            plt.ylabel("Position")
        elif i >= 2:
            plt.ylabel("Velocity")
            
    model_labels.append('Ground Truth')
    plt.figlegend(
        model_labels,
        loc='right',
        bbox_to_anchor=(1.05, 0.48),  # Adjust position here
        frameon=True
    )

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Shrink plot area to leave space