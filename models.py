import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np
import utils
import dataset


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
    def __init__(self):
        self.name = "Kalman Filter"
        self.A = None
        self.C = None
        self.W = None
        self.Q = None
        
    def train(self, X, Y):
        """Train the Kalman Filter

        Args:
            X (np array): Neural data, shape [timepoints, features]
            Y (np array): Neural data, shape [timepoints, kinematic outputs]
            
        Returns: 
            None
        """
        
        if self.A is not None: 
            raise ValueError("Tried to train_model a model that's already trained")
        
        ## I'm using an old version of a kalman filter I have, so the transposes/shapes are a bit janky
        Y = Y.T
        X = X.T
        
        YT = np.transpose(Y)
        
        ## Calculate A
        A = (Y[:, 1:] @ YT[:-1, :]) @ np.linalg.pinv(Y[:, 0:-1] @ YT[0:-1, :])
        ## Calculate C 
        C = X @ YT @ np.linalg.pinv(Y @ YT)
        
        # Find W 
        w = Y[:, 1:] - A @ Y[:, :-1]
        W = (w @ np.transpose(w)) / (np.shape(Y)[1] - 1)

        # Find Q
        q = X - C @ Y
        Q = (q @ np.transpose(q)) / (np.shape(Y)[1])
        
        self.A = A
        self.C = C
        self.W = W
        self.Q = Q
        
    def run(self, X, y_init):
        """Run the Kalman Filter

        Args:
            neural (np array): Neural data, shape [timepoints, features]
            y_init (np array): Neural data, shape [timepoints, kinematic outputs]
            
        Returns: 
            yhat (np array): Predictions, shape [timepoints, kinematic outputs]
        """
        if X.shape[0] < X.shape[1]:
            raise ValueError(f"X.shape[0] should be larger than X.shape[1], x shape:{X.shape}")
       
        X = X.T
        
        t = np.shape(X)[1]
        f = np.shape(y_init)[0]
        y_pred = np.empty([f, t])
        
        y_pred[:, 0] = y_init
            
        Pt = self.W
        for i in range(1, t):
            ylast = self.A @ y_pred[:, i-1]
            plast = self.A @ Pt @ self.A.T + self.W
            Kt = plast @ self.C.T @ np.linalg.pinv(self.C @ plast @ self.C.T + self.Q)
            y_pred[:, i] = ylast + Kt @(X[:, i] - self.C @ ylast)
            Pt = (np.eye(self.C.shape[1]) - Kt @ self.C) @ plast
            
        yhat = y_pred.T
        
        return yhat
    
  
class RidgeRegression(Model):
    def __init__(self, lbda = 'auto', intercept = True):
        self.name = "Ridge Regression"
        self.theta = None
        self.lbda = lbda
        self.intercept = intercept
        
    def train(self, X, Y):
        if self.theta is not None: 
            raise ValueError("Tried to train_model a model that's already trained")
        
        if self.lbda == 'auto':
            self.lbda = utils.get_lbda(X, Y, self.intercept)
            
        if self.intercept:
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self.theta, _, _, _ = np.linalg.lstsq(np.matmul(X.T, X) + self.lbda*np.eye(X.shape[1]), np.matmul(X.T, Y))
    
        
    def run(self, X, y_init):
        if X.shape[0] < X.shape[1]:
            raise ValueError(f"X.shape[0] should be larger than X.shape[1], x shape:{X.shape}")
       
        if self.intercept:
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
        
        yhat = np.matmul(X, self.theta)
        
        return yhat
        
   
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
    


         
        

         
        