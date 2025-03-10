from abc import ABC, abstractmethod
import numpy as np


class Model():
    def __init__():
        raise NotImplementedError
    
    @abstractmethod
    def train():
        raise NotImplementedError
    
    @abstractmethod
    def run():
        raise NotImplementedError
        
    
class KalmanFilter(Model):
    def train(self, neural, emg):
        """Train the Kalman Filter

        Args:
            neural (np array): Neural data, shape [timepoints, features]
            emg (np array): Neural data, shape [timepoints, kinematic outputs]
            
        Returns: 
            None
        """
        ## I'm using an old version of a kalman filter I have, so the transposes/shapes and variable names are a bit janky
        Y = neural.T
        X = emg.T
        X = XT
        
        XT = np.transpose(X)
        
        ## Calculate A
        a = X[:, 1:] @ XT[:-1, :]
        A = (X[:, 1:] @ XT[:-1, :]) @ np.linalg.pinv(X[:, 0:-1] @ XT[0:-1, :])
        ## Calculate C 
        C = Y @ XT @ np.linalg.pinv(X @ XT)
        
        # Find W 
        w = X[:, 1:] - A @ X[:, :-1]
        W = (w @ np.transpose(w)) / (np.shape(X)[1] - 1)

        # Find Q
        q = Y - C @ X
        Q = (q @ np.transpose(q)) / (np.shape(X)[1])
        
        self.A = A
        self.C = C
        self.W = W
        self.Q = Q
        
    def run_model(self, neural, y_init):
        """Run the Kalman Filter

        Args:
            neural (np array): Neural data, shape [timepoints, features]
            y_init (np array): Neural data, shape [timepoints, kinematic outputs]
            
        Returns: 
            yhat (np array): Predictions, shape [timepoints, kinematic outputs]
        """
        X = neural.T
        intl = y_init.T
        
        m = np.shape(intl)[0]
        k = np.shape(y_init)[1]
        X_corr = np.empty((m,k))
        
        X_corr[:, 0] = y_init
            
        Pt = self.W
        for i in range(1, k):
            xlast = self.A @ X_corr[:, i-1]
            plast = self.A @ Pt @ self.A.T + self.W
            # Kt = plast @ self.C.T @ utils.inverse_singular(self.C @ plast @ self.C.T + self.Q)
            Kt = plast @ self.C.T @ np.linalg.pinv(self.C @ plast @ self.C.T + self.Q)
            X_corr[:, i] = xlast +Kt @(Y[:, i] - self.C @ xlast)
            Pt = (np.eye(self.C.shape[1]) - Kt @ self.C) @ plast
            
        yhat = X_corr.T
        
        return yhat
    
class RidgeRegression(Model):
    def __init__(self, lamda):
        self.lamda = lamda
        
    def train(self, neural, emg):
        raise NotImplementedError
    
    def run_model(self, neural, y_init):
        raise NotImplementedError

class LSTM(Model):
    def __init__(self):
        raise NotImplementedError
        
    def train(self, neural, emg):
        raise NotImplementedError
    
    def run_model(self, neural, y_init):
        raise NotImplementedError
    
class MLP(Model):
    def __init__(self):
        raise NotImplementedError
        
    def train(self, neural, emg):
        raise NotImplementedError
    
    def run_model(self, neural, y_init):
        raise NotImplementedError
    
class KalmanNet(Model):
    def __init__(self):
        raise NotImplementedError
        
    def train(self, neural, emg):
        raise NotImplementedError
    
    def run_model(self, neural, y_init):
        raise NotImplementedError
    


         
        

         
        