import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from abc import ABC, abstractmethod
import numpy as np
import utils
import dataset


class Model():
    def __init__():
        raise NotImplementedError
    
    @abstractmethod
    def train_model():
        raise NotImplementedError
    
    @abstractmethod
    def run_model():
        raise NotImplementedError
        
    
class KalmanFilter(Model):
    def __init__(self):
        self.name = "Kalman Filter"
        self.A = None
        self.C = None
        self.W = None
        self.Q = None
        
    def train_model(self, X, Y):
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
        
    def run_model(self, X, y_init):
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
        if lbda == 1:
            self.name = "Linear Regression"
        else:
            self.name = f"Ridge Regression, lambda = {lbda}"
        self.theta = None
        self.lbda = lbda
        self.intercept = intercept
        
    def train_model(self, X, Y, plot = False):
        if self.theta is not None: 
            raise ValueError("Tried to train_model a model that's already trained")
        
        if self.lbda == 'auto':
            self.lbda = utils.get_lbda(X, Y, self.intercept, plot)
            self.name = f"Ridge Regression, lambda = {self.lbda}"
            
        if self.intercept:
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
             
        self.theta, _, _, _ = np.linalg.lstsq(np.matmul(X.T, X) + self.lbda*np.eye(X.shape[1]), np.matmul(X.T, Y))
    
        
    def run_model(self, X, y_init):
        if X.shape[0] < X.shape[1]:
            raise ValueError(f"X.shape[0] should be larger than X.shape[1], x shape:{X.shape}")
       
        if self.intercept:
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
        
        yhat = np.matmul(X, self.theta)
        
        return yhat
 
class LSTM(nn.Module):
    def __init__(self, hidden_size = 150, num_layers = 2, learning_rate = 1e-4, epochs = 10, batch_size = 64, device = 'cpu'):
        ''' 
        Initializes a LSTM

        Args:
            hidden_size:        size of hidden state in model 
            num_layers:         number of layers in model
            device:             optional, specifies what device to compute on. Default is cpu.
        Returns:
            None
        ''' 
        self.name = "LSTM"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        super().__init__()
    
    def train_model(self, X, Y, loss_fn = nn.MSELoss(), seed = 1, print_results = True, print_every = 2, plot = False):
        num_inputs = X.shape[1]
        num_outputs = Y.shape[1]

        X_train, Y_train, X_val, Y_val = dataset.get_val_set(X, Y)

        utils.set_seed(seed)

        # Initialize LSTM and linear layers
        self.lstm = nn.LSTM(
            input_size=num_inputs,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        ).to(self.device)
        
        self.linear = nn.Linear(self.hidden_size, num_outputs).to(self.device)
        
        # Define optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # Convert data to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y_train, dtype=torch.float32).to(self.device)
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        train_losses = []
        val_losses = []
        # Training loop
        for epoch in range(self.epochs):
            self.train()
            total_loss = 0.0
            for batch_X, batch_Y in train_loader:
                optimizer.zero_grad()
                
                # Add sequence dimension (assuming univariate time steps)
                batch_X = batch_X.unsqueeze(1)
                
                # Forward pass
                lstm_out, _ = self.lstm(batch_X)
                predictions = self.linear(lstm_out.squeeze(1))
                loss = loss_fn(predictions, batch_Y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            val_loss = None
            if len(X_val) > 0:
                self.eval()
                with torch.no_grad():
                    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    X_val_t = X_val_t.unsqueeze(1)
                    val_out, _ = self.lstm(X_val_t)
                    val_pred = self.linear(val_out.squeeze(1))
                    val_loss = loss_fn(val_pred, 
                                    torch.tensor(Y_val, dtype=torch.float32).to(self.device)).item()
            
            # Print results
            if print_results and (epoch % print_every == 0 or epoch == self.epochs-1):
                log = f"Epoch {epoch+1}/{self.epochs} | Train Loss: {total_loss/len(train_loader):.4f}"
                if val_loss is not None:
                    log += f" | Val Loss: {val_loss:.4f}"
                print(log)
            
            train_losses.append(total_loss/len(train_loader))
            val_losses.append(val_loss)
              
        if plot:   
            utils.plot_train_val_loss(train_losses,  val_losses)
        


    def run_model(self, X):
        if self.lstm is None or self.linear is None:
            raise RuntimeError("Model must be trained before inference")
            
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            X_tensor = X_tensor.unsqueeze(1)  # Add sequence dimension
            lstm_out, _ = self.lstm(X_tensor)
            predictions = self.linear(lstm_out.squeeze(1))
        return predictions.cpu().numpy()
    
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
    


         
        

         
        