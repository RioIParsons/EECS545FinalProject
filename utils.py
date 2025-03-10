import numpy as np
from sklearn.metrics import r2_score



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
        
        
def plot_predictions(y, yhat,  metric = corr()):
    raise NotImplementedError
    