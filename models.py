import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt

from abc import ABC, abstractmethod
import numpy as np
import utils
import dataset

import torch.nn.functional as F
import copy


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
        self.trained = False

        
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
        self.Kt_gains = []
        for i in range(1, t):
            ylast = self.A @ y_pred[:, i-1]
            plast = self.A @ Pt @ self.A.T + self.W
            Kt = plast @ self.C.T @ np.linalg.pinv(self.C @ plast @ self.C.T + self.Q)
            self.Kt_gains.append(np.linalg.norm(Kt))
            y_pred[:, i] = ylast + Kt @(X[:, i] - self.C @ ylast)
            Pt = (np.eye(self.C.shape[1]) - Kt @ self.C) @ plast
            
        yhat = y_pred.T
        
        return yhat     
    
class SSKalmanFilter(Model):
    def __init__(self):
        self.name = "Kalman Filter"
        self.A = None
        self.C = None
        self.W = None
        self.Q = None
        self.K = None
        
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
            # plast = self.A @ Pt @ self.A.T + self.W
            # Kt = plast @ self.C.T @ np.linalg.pinv(self.C @ plast @ self.C.T + self.Q)
            y_pred[:, i] = ylast + self.K @(X[:, i] - self.C @ ylast)
            # Pt = (np.eye(self.C.shape[1]) - Kt @ self.C) @ plast
            
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
        self.trained = False

        
    def train_model(self, X, Y, plot = False):
        if self.theta is not None: 
            raise ValueError("Tried to train_model a model that's already trained")
        
        if self.lbda == 'auto':
            self.lbda = utils.get_lbda(X, Y, intercept = self.intercept, plot = plot)
            self.name = f"Ridge Regression, lambda = {self.lbda}"
            
        if self.intercept:
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
             
        self.theta, _, _, _ = np.linalg.lstsq(np.matmul(X.T, X) + self.lbda*np.eye(X.shape[1]), np.matmul(X.T, Y))
    
        
    def run_model(self, X, y_init = None):
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
        self.trained = False
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
        


    def run_model(self, X, y_init = None):
        if self.lstm is None or self.linear is None:
            raise RuntimeError("Model must be trained before inference")
            
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            X_tensor = X_tensor.unsqueeze(1)  # Add sequence dimension
            lstm_out, _ = self.lstm(X_tensor)
            predictions = self.linear(lstm_out.squeeze(1))
        return predictions.cpu().numpy()
    
class MLP(nn.Module):
    def __init__(self, hidden_size = 150, num_layers = 2, learning_rate = 1e-4, epochs = 10, batch_size = 64, device = 'cpu'):
        ''' 
        Initializes a MLP

        Args:
            hidden_size:        size of hidden state in model 
            num_layers:         number of layers in model
            device:             optional, specifies what device to compute on. Default is cpu.
        Returns:
            None
        ''' 
        self.name = "MLP"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.trained = False
        super().__init__()
        
    def train_model(self, X, Y, loss_fn = nn.MSELoss(), seed = 1, print_results = True, print_every = 2, plot = False):
        num_inputs = X.shape[1]
        num_outputs = Y.shape[1]

        X_train, Y_train, X_val, Y_val = dataset.get_val_set(X, Y)
        utils.set_seed(seed)

        #Initialize the layers

        input_size = num_inputs
        output_size = num_outputs

        self.model = torch.nn.Sequential(

            # Hidden layer 1
            torch.nn.Linear(input_size, self.hidden_size),
            torch.nn.ReLU(),

            # Hidden layer 2
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),

            # Output layer
            torch.nn.Linear(self.hidden_size, output_size)
        ).to(self.device)


        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)

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
                #batch_X = batch_X.unsqueeze(1)
                
                # Forward pass
                predictions = self.model(batch_X)
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
                    #X_val_t = X_val_t.unsqueeze(1)
                    val_pred = self.model(X_val_t)
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
        


    def run_model(self, X, y_init = None):
            
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()
 
class KalmanNet(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, learning_rate=1e-3, epochs=10, batch_size=64, device=None, seq_len = 10):
        """
        Initializes a KalmanNet model. Adapted from Luis Cubillos

        Args:
            hidden_size: Size of hidden states in the GRU networks
            num_layers: Number of GRU layers
            learning_rate: Learning rate for Adam optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: Optional, specifies which device to compute on. Default is cuda if available, else cpu.
        """
        super().__init__()
        self.name = "KalmanNet"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.encoder_model = None
        
        ## ASK LUIS ABOUT
        self.gain_scaler_pos = 100
        self.gain_scaler_vel = 30000
        
        # Set device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Model components will be initialized during training
        self.A = None  # State transition matrix
        self.C = None  # Observation matrix
        self.W = None  # Process noise covariance
        self.Q = None  # Observation noise covariance
        
        # Neural network components
        self.lstm = None
        self.linear = None
        self.model = None  # Will hold the full KalmanNet model
        self.trained = False

    
    def train_model(self, X, Y, loss_fn=nn.MSELoss(), seed=1, print_results=True, print_every=1, plot = True):
        """
        Train the KalmanNet model

        Args:
            X: Neural data, shape [timepoints, features]
            Y: Kinematic data, shape [timepoints, output_dimensions]
            loss_fn: Loss function (default: MSELoss)
            seed: Random seed for reproducibility
            print_results: Whether to print training progress
            print_every: How often to print results (in epochs)
            
        Returns:
            None
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        optimizer, loader_train, loader_val = self.init_training(X, Y)

        metric = utils.corr()
        self._num_iteration = 0
        for epoch in range(self.epochs):
            for batch_idx, loader_dict in enumerate(loader_train):
                kin = loader_dict[1]
                emg = loader_dict[0]
                self._num_iteration += 1
                self.train()
                self.init_hidden(emg.shape[0])
                optimizer.zero_grad()
                ## ASK LUIS ABOUT INITIAL STATES
                states_hat, kgains = self.forward_batch(emg, kin[:, :, 0])
                states_hat = states_hat.to(self.device)

                # Extract the norm of the kalman gains
                kgains_norm = torch.norm(kgains, "fro", dim=(1, 2))
                
                loss = loss_fn(states_hat, kin)
                loss.backward()
                optimizer.step()
                
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
                # Correlation: {metric(states_hat.detach().cpu(), kin.detach().cpu()):.4f}, MSE: 

    def init_training(self, X, Y):
        X_tr, Y_tr, X_val, Y_val = dataset.get_val_set(X, Y)

        # # Convert data to PyTorch tensors
        # X_train = torch.tensor(X_tr, dtype=torch.float32).to(self.device)
        # Y_train = torch.tensor(Y_tr, dtype=torch.float32).to(self.device)
        
        # # Create DataLoader for batching
        # train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset, batch_size=self.batch_size, shuffle=True
        # )
        
        train_loader = utils.convert_to_torch_batches(X_tr, Y_tr, batch_size = self.batch_size, seq_len=self.seq_len, device = self.device, shuffle = True)
        val_loader = utils.convert_to_torch_batches(X_val, Y_val, batch_size = self.batch_size, seq_len=self.seq_len, device = self.device, shuffle = False)

        self.build_kf(X_tr, Y_tr)
        self.build_network()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer, train_loader, val_loader

    def build_kf(self, X, Y):
        """Initialize system dynamics matrices A, C, W, Q"""
        # Estimate state transition matrix A
        Y_prev = Y[:-1]
        Y_next = Y[1:]
        A = np.linalg.lstsq(Y_prev, Y_next, rcond=None)[0].T
        # A = torch.tensor(A, dtype=torch.float32).to(self.device)
        # A_T = self.A.T
        
        # Estimate observation matrix C
        C = np.linalg.lstsq(Y, X, rcond=None)[0].T
        # C = torch.tensor(self.C, dtype=torch.float32).to(self.device)
        # C_T = self.C.T
        
        # Estimate process noise covariance W
        Y_pred = Y_prev @ A.T
        residuals = Y_next - Y_pred
        W = residuals.T @ residuals / len(residuals)
        W = torch.tensor(W, dtype=torch.float32).to(self.device)
        
        # Estimate observation noise covariance Q
        X_pred = Y @ C.T
        residuals = X - X_pred
        Q = residuals.T @ residuals / len(residuals)
        Q = torch.tensor(Q, dtype=torch.float32).to(self.device)
        
        A = torch.tensor(A, dtype=torch.float32)

        self.A = A.to(self.device, non_blocking=True).float()
        self.A_T = torch.transpose(A, 0, 1).to(self.device, non_blocking=True).float()
        self.m = self.A.size()[0]

        C = torch.tensor(C, dtype=torch.float32).to(self.device)

        # Set Observation model
        self.C = C
        # self.C.Cs_to_device(self.device, non_blocking=True)
        self.n = self.C.shape[0]
        ##ASK LUIS ABOUT n

        # Noise covariances
        self.W = W.to(self.device, non_blocking=True).float() if W is not None else None
        self.Q = Q.to(self.device, non_blocking=True).float() if Q is not None else None

    def build_network(self):
        # Initialize hidden states
        self.prior_Q = torch.eye(self.m).to(self.device)
        self.prior_Sigma = torch.zeros((self.m, self.m)).to(self.device)
        self.prior_S = torch.eye(self.n).to(self.device)

        # GRU to track Q
        # Represents the size of the output of the linear layers.
        in_mult_knet = 5
        self.d_input_Q = self.m * in_mult_knet
        self.d_hidden_Q = self.m**2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * in_mult_knet
        self.d_hidden_Sigma = self.m**2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)

        # GRU to track S
        self.d_input_S = self.n**2 + 2 * self.n * in_mult_knet
        self.d_hidden_S = self.n**2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)

        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n**2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1), nn.ReLU()
        ).to(self.device)

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * in_mult_knet
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2),
        ).to(self.device)

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m**2
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3), nn.ReLU()
        ).to(self.device)

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4), nn.ReLU()
        ).to(self.device)

        # Fully connected 5
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * in_mult_knet
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5), nn.ReLU()
        ).to(self.device)

        # Fully connected 6
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * in_mult_knet
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6), nn.ReLU()
        ).to(self.device)

        # Fully connected 7
        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * in_mult_knet
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7), nn.ReLU()
        ).to(self.device)

    def init_hidden(self, batch_size):
        self.batch_size = batch_size
        # if not self.config["reg_kf"]["run"]:
        #     # weight = next(self.parameters()).data
        #     # # hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        #     # hidden = (weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device),
        #     #             weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device))
        #     # self.hn = hidden

        weight = next(self.parameters()).data
        hidden = weight.new(1, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S = (
            self.prior_S.flatten().reshape(1, 1, -1).repeat(1, self.batch_size, 1)
        )  # batch size expansion
        hidden = weight.new(1, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma = (
            self.prior_Sigma.flatten()
            .reshape(1, 1, -1)
            .repeat(1, self.batch_size, 1)
        )  # batch size expansion
        hidden = weight.new(1, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q = (
            self.prior_Q.flatten().reshape(1, 1, -1).repeat(1, self.batch_size, 1)
        )  # batch size expansion
        
    def forward_batch(
        self, input_batch, initial_state, return_kgain=False, ground_truth_batch=None
    ):
        
        # y_batch must be: (batch_size, n, seq_len)
        input_batch = input_batch.to(self.device, non_blocking=True)
        ground_truth_batch = (
            ground_truth_batch.to(self.device, non_blocking=True)
            if ground_truth_batch is not None
            else None
        )
        states_out = torch.empty(input_batch.shape[0], self.m, input_batch.shape[2]).to(self.device)
        states_out[:, :, 0] = initial_state
        kgains = torch.empty(
            input_batch.shape[0], self.m, self.n, input_batch.shape[2])
        
        self.init_sequence(initial_state.unsqueeze(-1))
        for t in range(1, input_batch.shape[-1]):
            states_out[:, :, t] = self.forward(
                input_batch[:, :, t].unsqueeze(-1),
                ground_truth=(
                    ground_truth_batch[:, :, t].unsqueeze(-1)
                    if ground_truth_batch is not None
                    else None
                ),
            )

            # # Initialize sequence and then pass forward the rest of it
            # states_out[b, :, 1:] = self.forward_sequence(y_batch[b, :, 1:])
            kgains[:, :, :, t] = self.k_gain
        return states_out, kgains

    def init_sequence(self, initial_state):
        # initial_state: (batch, m,1)
        # Both prior and posterior are the same at the beginning
        self.state_prior = initial_state.to(self.device, non_blocking=True)
        self.state_posterior = initial_state.to(self.device, non_blocking=True)
        self.state_prev_posterior = self.state_posterior.clone()
        # self.last_y = torch.zeros(self.n).to(self.device, non_blocking=True)
        
        ## ASK LUIS ABOUT C AS METHOD OR MATRIX
        C_expanded = self.C.unsqueeze(0).expand(self.state_posterior.shape[0], -1, -1)  # [64, 16, 2]

        self.last_input = torch.bmm(C_expanded, self.state_posterior).to(
            self.device, non_blocking=True
        )
        # Starting value of P is the state noise covariance
        self.P = torch.clone(self.W) if self.W is not None else None
        
    def forward(self, input, ground_truth=None):
        # input must be: (batch, n, 1)
        input = input.to(self.device, non_blocking=True)
        # Apply encoder model to input if applicable.
        if self.encoder_model is not None:
            # FIXME: figure it out for different input sizes
            input = self.encoder_model(input.permute(0, 2, 1)).permute(0, 2, 1)
        return self.kf_step(input, ground_truth=ground_truth)
   
    def kf_step(self, input, ground_truth=None):
        # Compute Priors
        self.step_prior()

        #POTISS
        # Compute Kalman Gain
        # if self.config["reg_kf"]["run"]:
        # self.k_gain = self.kgain_regkf(ground_truth=ground_truth)
        # else:
        self.k_gain = self.kgain_network(input)

        # Compute the 1-st posterior moment
        delta_input = input - self.input_prior
        innovation = torch.bmm(self.k_gain, delta_input.float())
        self.state_prev_posterior = self.state_posterior.clone()
        self.state_posterior = self.state_prior + innovation
      
        self.last_input = input
        return torch.squeeze(self.state_posterior)
    
    def step_prior(self):
        # Prior state
        self.x_prev_prior = self.state_prior
        # FIXME: move this so that it only runs once.
        A_batch = self.A.expand(self.x_prev_prior.shape[0], -1, -1)
        self.state_prior = torch.bmm(A_batch, self.state_posterior)

        # Observation prior (channels)
        C_expanded = self.C.unsqueeze(0).expand(self.state_posterior.shape[0], -1, -1)  # [64, 16, 2]
        self.input_prior = torch.bmm(C_expanded, self.state_prior)

    def kgain_network(self, input):
        # Input to Kalman Gain Network
        obs_diff, obs_innov_diff, fw_update_diff, fw_evol_diff = self.compute_features(
            input
        )

        # Kalman Gain Network Step
        KG = self.network_fwd_pass(
            obs_diff, obs_innov_diff, fw_update_diff, fw_evol_diff
        )

        # Reshape Kalman Gain to a Matrix
        KG = torch.reshape(KG.squeeze(), (self.batch_size, self.m, self.n))
        # Scale Kalman Gain
        KG = KG / self.gain_scaler_pos
        # KG[:, vel_idx, :] = KG[:, vel_idx, :] / self.gain_scaler_vel
        # Add row of zeros
        # KG = torch.cat(
        #     [KG, torch.zeros(self.batch_size, 1, self.n).to(self.device)], dim=1
        # )
        return KG
    
    def compute_features(self, input):
        #  Feature 1: yt - yt-1
        obs_diff = (
            input - self.last_input
            if self.last_input is not None
            else torch.zeros(self.n).to(self.device)
        )
        # Feature 2: yt - y_t+1|t
        obs_innov_diff = input - self.input_prior
        # Feature 4: x_t-1|t-1 - x_t-1|t-2
        fw_update_diff = (
            self.state_posterior[:, : self.m, :] - self.x_prev_prior[:, : self.m, :]
        )
        fw_evol_diff = (
            self.state_posterior[:, : self.m, :]
            - self.state_prev_posterior[:, : self.m, :]
        )

        # Normalize features
        obs_diff = F.normalize(obs_diff, p=2, dim=1, eps=1e-12, out=None).squeeze(-1)
        obs_innov_diff = F.normalize(
            obs_innov_diff, p=2, dim=1, eps=1e-12, out=None
        ).squeeze(-1)
        fw_update_diff = F.normalize(
            fw_update_diff, p=2, dim=1, eps=1e-12, out=None
        ).squeeze(-1)
        fw_evol_diff = F.normalize(
            fw_evol_diff, p=2, dim=1, eps=1e-12, out=None
        ).squeeze(-1)

        # Network input
        return obs_diff, obs_innov_diff, fw_update_diff, fw_evol_diff

    def network_fwd_pass(self, obs_diff, obs_innov_diff, fw_update_diff, fw_evol_diff):
        # def expand_dim(x):
        #     expanded = torch.empty(1, self.batch_size, x.shape[-1]).to(self.device)
        #     expanded[0, :, :] = x
        #     return expanded

        obs_diff = obs_diff.unsqueeze(0)
        obs_innov_diff = obs_innov_diff.unsqueeze(0)
        fw_evol_diff = fw_evol_diff.unsqueeze(0)
        fw_update_diff = fw_update_diff.unsqueeze(0)

        ####################
        ### Forward Flow ###
        ####################

        # FC 5
        in_FC5 = fw_evol_diff
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5
        self.out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)

        # FC 6
        in_FC6 = fw_update_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((self.out_Q, out_FC6), 2)
        self.out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC 1
        in_FC1 = self.out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)

        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        self.out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        # FC 2
        in_FC2 = torch.cat((self.out_Sigma, self.out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((self.out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((self.out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4

        return out_FC2

    def run_model(self, X, y_init, return_kgains=False):
        # Chans (t, n), states (t, m + 1)
        # Add bias term
        
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        initial_state = torch.tensor(y_init, dtype=torch.float32).to(self.device).unsqueeze(0).T
        # Initialize hidden state with batch size 1 if appropriate
        # if not self.config["reg_kf"]["run"]:
        self.init_hidden(1)
        ground_truth_batched = None
        # else:
        # ground_truth_batched = states.T.unsqueeze(0)
        # Save the kgains for the test to return them
        states_hat, kgains = self.forward_batch(
            X.T.unsqueeze(0),
            initial_state.T,
            ground_truth_batch=ground_truth_batched,
        )
        # remove batch, get back to (t,m+1)
        if return_kgains:
            # kgains returnes as (time, channels, states)
            return states_hat.squeeze().T.detach().cpu().numpy(), kgains.squeeze().permute(2, 1, 0).detach().cpu().numpy()
        else:
            return states_hat.squeeze().T.detach().cpu().numpy()
        
        