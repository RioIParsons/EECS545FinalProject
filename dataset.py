import os
import pickle
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_data(fpath = 'Z_Joker_2025-01-09_Run-002.mat', downsample = 10, velocity = True, lag = None):
    """Load the chosed data

    Returns:
        EMG_continuous: EMG data, shape [timepoints, features]
        kin_continuous: Kinematics, shape [timepoints, kinematic dimensions]
    """

    if fpath[-3:] == "mat":
        mat = scipy.io.loadmat(fpath)
        dt = .001

        inds = mat['z']['TrialSuccess'][0, :].astype(bool)
        
        kin_trials = mat['z'][0, :]['FingerAnglesTIMRL'][inds] #shape (n_trials, ), filtering out unsuccessful trials
        kin_continuous = np.vstack(kin_trials)[:, [1, 3]] 
    
        EMG_trials = mat['z'][0, :]['NeuralFeature'][inds]
        EMG_continuous = np.vstack(EMG_trials)[:, :16]
    elif fpath[-3:] == "npy":
        ds = np.load(fpath)
    elif fpath[-3:] == "pkl":
        dpath = os.path.join('dataset_60.pkl')
        
        with open(dpath, 'rb') as f:
            EMG_continuous, kinematics, _, _ = pickle.load(f)
            kin_continuous = kinematics[:, :2]
    
    if downsample is not None: 
        t = EMG_continuous.shape[0]
        EMG_continuous = EMG_continuous[:t // downsample * downsample].reshape(-1, downsample, EMG_continuous.shape[1]).mean(axis=1)
        kin_continuous = kin_continuous[:t // downsample * downsample].reshape(-1, downsample, kin_continuous.shape[1]).mean(axis=1)

    if velocity:
        vel = np.diff(kin_continuous, axis = 0)/dt
        kin_continuous = np.hstack([kin_continuous[1:, :], vel])
        EMG_continuous = EMG_continuous[1:, :]
        
    if lag is not None:
        kin_continuous = kin_continuous[lag:, :]
        EMG_continuous = EMG_continuous[:-lag, :]

    
    assert EMG_continuous.shape[0] == kin_continuous.shape[0]

    return EMG_continuous, kin_continuous

def partition_data(X, Y, split_ratio = [.8, .2], normalize = True):
    assert np.sum(split_ratio) == 1
    
    train_idx = int(X.shape[0] * split_ratio[0])
    x_train, y_train = X[:train_idx, :], Y[:train_idx, :]
    x_test, y_test = X[train_idx:, :], Y[train_idx:, :]
    
    if normalize: 
        x_train_mean = x_train.mean(axis = 0)
        x_train_std = np.std(x_train, axis=0)

        y_train_mean = np.mean(y_train, axis=0)
        y_train_std = np.std(y_train, axis=0)   
             
        x_train = (x_train - x_train_mean) / (x_train_std + 1e-14)
        x_test = (x_test - x_train_mean) / (x_train_std + 1e-14)

        y_train = (y_train - y_train_mean) / (y_train_std + 1e-10)
        y_test = (y_test - y_train_mean) / (y_train_std + 1e-10)

        
    return x_train, y_train, x_test, y_test

def get_val_set(X, Y, ratio = [.875, .125]):
    train_idx = int(X.shape[0] * (ratio[0]))
    x_train, y_train  = X[:train_idx, :], Y[:train_idx, :]
    x_val, y_val = X[train_idx:, :], Y[train_idx:, :]
    
    return x_train, y_train, x_val, y_val


    