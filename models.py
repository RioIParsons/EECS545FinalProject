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
        super().__init__()
    
        
    def train_model(self, X, Y, loss_fn = nn.MSELoss(), seed = 1, print_results = True, print_every = 2, plot = True):
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
        


    def run_model(self, X):
            
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()
    
class KalmanNet(Model):
    def __init__(self, hidden_size=64, num_layers=1, learning_rate=1e-3, epochs=10, batch_size=64, device=None):
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
        self.name = "KalmanNet"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
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
        
    def _build_model(self, input_dim, state_dim):
        """Builds the KalmanNet model structure"""
        in_mult_knet = 5
        
        # GRU to track Q
        d_input_Q = state_dim * in_mult_knet
        d_hidden_Q = state_dim**2
        GRU_Q = nn.GRU(d_input_Q, d_hidden_Q).to(self.device)
        
        # GRU to track Sigma
        d_input_Sigma = d_hidden_Q + state_dim * in_mult_knet
        d_hidden_Sigma = state_dim**2
        GRU_Sigma = nn.GRU(d_input_Sigma, d_hidden_Sigma).to(self.device)
        
        # GRU to track S
        n = input_dim  # Observation dimension
        d_input_S = n**2 + 2 * n * in_mult_knet
        d_hidden_S = n**2
        GRU_S = nn.GRU(d_input_S, d_hidden_S).to(self.device)
        
        # Feature extraction networks
        FC5 = nn.Sequential(
            nn.Linear(state_dim, state_dim * in_mult_knet), 
            nn.ReLU()
        ).to(self.device)
        
        FC6 = nn.Sequential(
            nn.Linear(state_dim, state_dim * in_mult_knet), 
            nn.ReLU()
        ).to(self.device)
        
        FC7 = nn.Sequential(
            nn.Linear(2 * n, 2 * n * in_mult_knet), 
            nn.ReLU()
        ).to(self.device)
        
        # Processing networks
        FC1 = nn.Sequential(
            nn.Linear(d_hidden_Sigma, n**2), 
            nn.ReLU()
        ).to(self.device)
        
        FC2 = nn.Sequential(
            nn.Linear(d_hidden_S + d_hidden_Sigma, (d_hidden_S + d_hidden_Sigma) * in_mult_knet),
            nn.ReLU(),
            nn.Linear((d_hidden_S + d_hidden_Sigma) * in_mult_knet, n * state_dim),
        ).to(self.device)
        
        FC3 = nn.Sequential(
            nn.Linear(d_hidden_S + n * state_dim, state_dim**2), 
            nn.ReLU()
        ).to(self.device)
        
        FC4 = nn.Sequential(
            nn.Linear(d_hidden_Sigma + state_dim**2, d_hidden_Sigma), 
            nn.ReLU()
        ).to(self.device)
        
        return {
            'GRU_Q': GRU_Q,
            'GRU_Sigma': GRU_Sigma,
            'GRU_S': GRU_S,
            'FC1': FC1,
            'FC2': FC2,
            'FC3': FC3,
            'FC4': FC4,
            'FC5': FC5,
            'FC6': FC6,
            'FC7': FC7,
            'd_hidden_Q': d_hidden_Q,
            'd_hidden_Sigma': d_hidden_Sigma,
            'd_hidden_S': d_hidden_S,
        }
    
    def _init_system_dynamics(self, X, Y):
        """Initialize system dynamics matrices A, C, W, Q"""
        # Estimate state transition matrix A
        Y_prev = Y[:-1]
        Y_next = Y[1:]
        self.A = np.linalg.lstsq(Y_prev, Y_next, rcond=None)[0].T
        self.A = torch.tensor(self.A, dtype=torch.float32).to(self.device)
        self.A_T = self.A.T
        
        # Estimate observation matrix C
        self.C = np.linalg.lstsq(Y, X, rcond=None)[0].T
        self.C = torch.tensor(self.C, dtype=torch.float32).to(self.device)
        self.C_T = self.C.T
        
        # Estimate process noise covariance W
        Y_pred = Y_prev @ self.A.cpu().numpy().T
        residuals = Y_next - Y_pred
        self.W = residuals.T @ residuals / len(residuals)
        self.W = torch.tensor(self.W, dtype=torch.float32).to(self.device)
        
        # Estimate observation noise covariance Q
        X_pred = Y @ self.C.cpu().numpy().T
        residuals = X - X_pred
        self.Q = residuals.T @ residuals / len(residuals)
        self.Q = torch.tensor(self.Q, dtype=torch.float32).to(self.device)
    
    def _init_hidden(self, batch_size, model_components):
        """Initialize hidden states for the GRUs"""
        # Create properly sized hidden states directly
        h_Q = torch.eye(self.state_dim).flatten().unsqueeze(0).repeat(1, batch_size, 1).to(self.device)
        h_Sigma = torch.zeros(1, batch_size, self.state_dim**2).to(self.device)
        h_S = torch.eye(self.input_dim).flatten().unsqueeze(0).repeat(1, batch_size, 1).to(self.device)
        
        return h_Q, h_Sigma, h_S
    
    def _kf_step(self, x, state_prior, input_prior, state_posterior, model_components, h_Q, h_Sigma, h_S):
        """
        Performs one KalmanNet filtering step

        Args:
            x: Current observation
            state_prior: Prior state estimate
            input_prior: Prior observation estimate
            state_posterior: Previous posterior state estimate
            model_components: Dictionary of KalmanNet components
            h_Q, h_Sigma, h_S: Hidden states for GRUs
            
        Returns:
            state_posterior: Updated state estimate
            h_Q, h_Sigma, h_S: Updated hidden states
        """
        # Feature extraction
        obs_diff = x - input_prior
        obs_diff = nn.functional.normalize(obs_diff, p=2, dim=1, eps=1e-12)

        fw_evol_diff = state_posterior - state_prior
        fw_evol_diff = nn.functional.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)

        # Process through feature networks - shape: [batch_size, state_dim*in_mult_knet]
        out_FC5 = model_components['FC5'](fw_evol_diff)

        # For GRU input, shape should be [seq_len=1, batch_size, features]
        out_FC5 = out_FC5.unsqueeze(0)  

        # Process through Q-GRU
        out_Q, h_Q = model_components['GRU_Q'](out_FC5, h_Q)

        # Feature for sigma
        state_diff = state_posterior - state_prior
        state_diff = nn.functional.normalize(state_diff, p=2, dim=1, eps=1e-12)
        out_FC6 = model_components['FC6'](state_diff)
        out_FC6 = out_FC6.unsqueeze(0)  # [1, batch_size, features]

        # Process through Sigma-GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, h_Sigma = model_components['GRU_Sigma'](in_Sigma, h_Sigma)

        # Process for S-GRU
        out_FC1 = model_components['FC1'](out_Sigma)

        # Feature for S
        in_FC7 = torch.cat((obs_diff, obs_diff), 1)  # Using obs_diff twice
        out_FC7 = model_components['FC7'](in_FC7)
        out_FC7 = out_FC7.unsqueeze(0)  # [1, batch_size, features]

        # Process through S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, h_S = model_components['GRU_S'](in_S, h_S)

        # Compute Kalman gain
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        K_gain = model_components['FC2'](in_FC2)

        # Reshape K_gain to have correct dimensions for matrix multiplication
        # The shape should be [batch_size, state_dim, input_dim]
        K_gain = K_gain.view(K_gain.shape[1], self.state_dim, self.input_dim)

        # Apply Kalman gain to innovation
        # Need to add dimension for matrix multiplication
        obs_diff_expanded = obs_diff.unsqueeze(2)  # [batch_size, input_dim, 1]

        # Result will be [batch_size, state_dim, 1]
        innovation = torch.bmm(K_gain, obs_diff_expanded)

        # Remove the last dimension to match state dimensions
        innovation = innovation.squeeze(2)  # [batch_size, state_dim]

        # Update state
        state_posterior = state_prior + innovation

        # Update hidden state for Sigma-GRU
        # Reshape K_gain for concatenation
        K_gain_flat = K_gain.view(1, K_gain.shape[0], -1)

        in_FC3 = torch.cat((out_S, K_gain_flat), 2)
        out_FC3 = model_components['FC3'](in_FC3)

        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = model_components['FC4'](in_FC4)
        h_Sigma = out_FC4

        return state_posterior, h_Q, h_Sigma, h_S
    
    def train_model(self, X, Y, loss_fn=nn.MSELoss(), seed=1, print_results=True, print_every=1):
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
        
        # Store dimensions
        self.input_dim = X.shape[1]  # Number of neural features
        self.state_dim = Y.shape[1]  # Number of kinematic outputs
        
        # Init system dynamics (A, C, W, Q matrices)
        self._init_system_dynamics(X, Y)
        
        # Build KalmanNet components
        self.model_components = self._build_model(self.input_dim, self.state_dim)
        
        # Convert data to PyTorch tensors
        X_train = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y, dtype=torch.float32).to(self.device)
        
        # Create DataLoader for batching
        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Get all model parameters
        params = []
        for _, component in self.model_components.items():
            if isinstance(component, nn.Module):
                params.extend(list(component.parameters()))
        
        # Create optimizer
        optimizer = torch.optim.Adam(params, lr=self.lr)
        
        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_X, batch_Y in train_loader:
                batch_size = batch_X.shape[0]
                
                # Initialize hidden states
                h_Q, h_Sigma, h_S = self._init_hidden(batch_size, self.model_components)
                
                # Initialize state for first time step
                state_posterior = batch_Y[0].unsqueeze(0).repeat(batch_size, 1)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Total loss for this batch
                batch_loss = 0
                
                # Process each time step
                for t in range(1, min(100, len(batch_X))):  # Limit sequence length to avoid memory issues
                    # Compute prior
                    state_prior = torch.matmul(self.A, state_posterior.unsqueeze(-1)).squeeze(-1)
                    input_prior = torch.matmul(self.C, state_prior.unsqueeze(-1)).squeeze(-1)
                    
                    # KF step
                    state_posterior, h_Q, h_Sigma, h_S = self._kf_step(
                        batch_X[t], 
                        state_prior, 
                        input_prior,
                        state_posterior, 
                        self.model_components,
                        h_Q, h_Sigma, h_S
                    )
                    
                    # Compute loss
                    step_loss = loss_fn(state_posterior, batch_Y[t])
                    batch_loss += step_loss
                
                # Backward pass and optimization
                batch_loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                batch_count += 1
            
            if print_results and (epoch % print_every == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/batch_count:.4f}")
    
    def run_model(self, X, y_init):
        """
        Run the trained KalmanNet model

        Args:
            X: Neural data, shape [timepoints, features]
            y_init: Initial state, shape [output_dimensions]
            
        Returns:
            Y_pred: Predicted states, shape [timepoints, output_dimensions]
        """
        if self.model_components is None or self.A is None:
            raise RuntimeError("Model must be trained before inference")
        
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_init_tensor = torch.tensor(y_init, dtype=torch.float32).to(self.device)
        
        # Prepare output array
        predictions = np.zeros((X.shape[0], self.state_dim))
        predictions[0] = y_init
        
        # Set model components to evaluation mode
        for component_name, component in self.model_components.items():
            if isinstance(component, nn.Module):
                component.eval()
        
        # Initialize hidden states for a single sequence
        h_Q, h_Sigma, h_S = self._init_hidden(1, self.model_components)
        
        # Initialize state
        state_posterior = y_init_tensor.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            for t in range(1, len(X_tensor)):
                # Compute prior
                state_prior = torch.matmul(self.A, state_posterior.unsqueeze(-1)).squeeze(-1)
                input_prior = torch.matmul(self.C, state_prior.unsqueeze(-1)).squeeze(-1)
                
                # KF step
                state_posterior, h_Q, h_Sigma, h_S = self._kf_step(
                    X_tensor[t].unsqueeze(0),  # Add batch dimension
                    state_prior,
                    input_prior,
                    state_posterior,
                    self.model_components,
                    h_Q, h_Sigma, h_S
                )
                
                # Store prediction
                predictions[t] = state_posterior.cpu().numpy()
        
        return predictions
    


         
        

         
        