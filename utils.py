import copy
import random
import torch
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import dataset
import models
from sklearn.model_selection import TimeSeriesSplit, KFold


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

## Eval/Perf fxnx    
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

def plot_train_val_loss(train_losses,  val_losses):
    plt.figure()
    plt.plot(train_losses, label = "Train Losses")
    plt.plot(val_losses, label = "Val Losses")
    plt.legend()
    plt.ylabel("MSE loss")
    plt.xlabel("Epochs")

def compare_perfs(Y, yhats, model_labels, colors, start = 0, end = -1):
    labels = ['First Finger\nPosition', 'Second Finger\nPosition']
    ndims = 2

    nplots = len(yhats)
    assert(nplots == len(model_labels))
    
    fig = plt.figure(figsize = [17, 10])
    plt.suptitle("Comparison of Model Performances", fontsize=20, fontweight='bold')
    metric = corr()
     
    for i in range(nplots):
        for j in range(ndims):
            # print(f"i: {i*2}, j: {j%2}, total: {(i*2)+1+j%2}")
            plt.subplot(nplots, 2, (i*2)+1+j%2) 
        
            plt.plot(yhats[i][start:end, j], color = colors[i], label = model_labels[i] )
                
            plt.plot(Y[start:end, j], color = 'black', label = "Ground Truth")

            plt.title(model_labels[i], fontsize = 17, fontweight='bold')
          
            if i == nplots-1:
                plt.xlabel("Timepoint", fontsize = 15)
            # plt.ylabel(labels[j],  fontsize = 15)
            plt.xlim([0, end-start])
            ax =plt.gca()
            ylims = ax.get_ylim()
            plt.annotate(f"Correlation: {round(metric(yhats[i][start:end, j], Y[start:end, j]), 4)}", xy = [end-start-250, ylims[0]+.25])

    plt.subplots_adjust(wspace=5)  # Increase this value for more space

    fig.text(0.02, 0.5, 'First Finger Position', va='center', rotation='vertical', fontsize = 15)
    fig.text(.5, 0.5, 'Second Finger Position', va='center', rotation='vertical', fontsize = 15)

    plt.tight_layout(rect=[0.03, 0, 1, 0.95])  # Shrink plot area to leave space
    plt.savefig("res_fig.svg")
  
def cross_corr_box_plot(X, Y, model_list, labels, k = 5):

    X, Y, _, _ = dataset.partition_data(X, Y, split_ratio=[1, 0], normalize = True) #This just normalizes X and Y to prevent future issues

    assert (len(model_list) == len(labels))
    fold_size = X.shape[0] // 5
   
    metric = corr()
    corrs = [[] for _ in range(len(model_list))]
    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k - 1 else X.shape[0]
        test_idx = np.arange(test_start, test_end)
        train_idx = np.setdiff1d(np.arange(X.shape[0]), test_idx)
        print(f"train: {train_idx}, {len(train_idx)}")
        print(f"test: {test_idx}, {len(test_idx)}")

        X_train_fold, X_test_fold = X[train_idx, :], X[test_idx, :]
        Y_train_fold, Y_test_fold = Y[train_idx, :], Y[test_idx, :]

        for j, model in enumerate(model_list):
            kf = copy.deepcopy(model)

            kf.train_model(X_train_fold, Y_train_fold)
            Y_hat_fold = kf.run_model(X_test_fold, Y_test_fold[0,:])
            current_corr = np.array(metric(Y_hat_fold, Y_test_fold)).mean()
            corrs[j].append(current_corr)

    plt.figure(figsize=(8, 6))
    plt.boxplot(corrs, labels=labels)
    plt.title('Cross-Validation Correlation Scores')
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Model')
    plt.grid(True)
    plt.savefig("bplot.svg")

def k_gains_plot(kf_gains, knet_gains, ytest, start = 0, end = 500):
    gains = list(knet_gains)
    frobenius_norms = [np.linalg.norm(x, ord = 'fro') for x in gains]

    plt.figure(figsize=(12, 8))
    plt.plot(frobenius_norms[start:end], label='Kalman Gain from KalmanNet', color='blue', alpha=0.7)
    plt.plot(np.array(kf_gains[start:end]), label='Kalman Gain from Kalman Filter', color='red')
    plt.title('Kalman Gain Comparison')
    plt.plot(ytest[100:500,0], label='Index Kinematics', color='black')
    plt.xlabel('Time Step')
    plt.ylabel('Kalman Gain Frobenius Norm')
    plt.legend()
    plt.grid(True)
    plt.xlim([0, end-start])
    plt.tight_layout()
    plt.savefig("kgains.svg")

## True Utils  
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
        
def get_lbda(X, Y, intercept, lbda_start = 0, lbda_end = 5, lbda_step = 5, metric = corr(), plot = False):
    
    X_train, Y_train, X_val, Y_val = dataset.get_val_set(X, Y)
    
    lbdas = np.arange(lbda_start, lbda_end + lbda_step, lbda_step)
    perfs = np.empty([len(lbdas), Y_train.shape[1]])
    for i, lbda in enumerate(lbdas): 
        model = models.RidgeRegression(lbda, intercept)
        model.train_model(X_train, Y_train)
        yhat = model.run_model(X_val, None)
        perfs[i, :] = metric(Y_val, yhat)
        
    if plot:
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
   
def calc_corr_torch(x,xhat):
    """
    Compute the correlation between two tensors.
    """
    if len(x.shape) < 3:
        x = x[None, :, :]
        xhat = xhat[None, :, :]
    corr = torch.zeros([x.shape[0], x.shape[1]])
    for j in range(x.shape[0]):
        for i in range(x.shape[1]):
            corr[j, i] = torch.corrcoef(torch.stack((x[j, i, :], xhat[j, i, :])))[0,1]
    return torch.squeeze(corr)
        
def convert_to_torch_batches(X, Y, batch_size, seq_len, device, shuffle):
    
    total_len = (X.shape[0] // seq_len) * seq_len
    X = X[:total_len]
    Y = Y[:total_len]

    X = X.reshape(-1, seq_len, X.shape[1])
    Y = Y.reshape(-1, seq_len, Y.shape[1])
    X_ten = torch.tensor(X, dtype=torch.float32).to(device).transpose(1, 2)
    Y_ten = torch.tensor(Y, dtype=torch.float32).to(device).transpose(1, 2)
    
    # Create DataLoader for batching
    dataset = torch.utils.data.TensorDataset(X_ten, Y_ten)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
    
    return loader
