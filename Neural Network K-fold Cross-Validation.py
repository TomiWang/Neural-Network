# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 00:12:39 2025

@author: tomiz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
import tensorflow as tf
from tensorflow.keras import layers, models

def load_and_preprocess_data(file_path):
    file_path = input("Enter the file path: ")
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    reflectance_cols = [col for col in df.columns if col.startswith("wavelength")]
    abundance_cols = [col for col in df.columns if col.startswith("abundance")]
    scaler_ref = MinMaxScaler()
    df[reflectance_cols] = scaler_ref.fit_transform(df[reflectance_cols])
    scaler_ab = MinMaxScaler()
    df[abundance_cols] = scaler_ab.fit_transform(df[abundance_cols])
    return df, reflectance_cols, abundance_cols

def data_augmentation(df, reflectance_cols, noise_std=0.01, shift=0.005):
    augmented_df = df.copy()
    noise = np.random.normal(0, noise_std, size=augmented_df[reflectance_cols].shape)
    augmented_df[reflectance_cols] = augmented_df[reflectance_cols] + noise + shift
    augmented_df[reflectance_cols] = augmented_df[reflectance_cols].clip(0, 1)
    return augmented_df

def linear_mixing_model(S, a):
    return np.dot(S, a)

def train_baseline_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def build_nn_model(input_dim, output_dim):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_kernel_model(X_train, y_train):
    kr_model = KernelRidge(kernel='rbf')
    kr_model.fit(X_train, y_train)
    return kr_model

def k_fold_cross_validation(model_func, X, y, k=5, epochs=50, is_nn=False):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = []
    mae_scores = []
    fold = 1
    for train_index, val_index in kf.split(X):
        print(f"Fold {fold}:")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        if is_nn:
            model = model_func(X_train.shape[1], y_train.shape[1])
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=0)
            y_pred = model.predict(X_val)
            plt.figure()
            plt.plot(history.history['loss'], label='train loss')
            plt.plot(history.history['val_loss'], label='val loss')
            plt.title(f"NN Training Loss - Fold {fold}")
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.legend()
            plt.show()
        else:
            model = model_func(X_train, y_train)
            y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        mse_scores.append(mse)
        mae_scores.append(mae)
        print(f"Fold {fold} - MSE: {mse:.4f}, MAE: {mae:.4f}\n")
        fold += 1
    return mse_scores, mae_scores

def train_and_evaluate(X, y, model_choice='nn', k=5, epochs=50):
    if model_choice == 'baseline':
        print("Training Baseline (Linear Regression) Model with K-Fold Cross Validation")
        mse_scores, mae_scores = k_fold_cross_validation(train_baseline_model, X, y, k=k, epochs=epochs, is_nn=False)
    elif model_choice == 'nn':
        print("Training Neural Network Model with K-Fold Cross Validation")
        mse_scores, mae_scores = k_fold_cross_validation(build_nn_model, X, y, k=k, epochs=epochs, is_nn=True)
    elif model_choice == 'kernel':
        print("Training Kernel Ridge Regression Model with K-Fold Cross Validation")
        mse_scores, mae_scores = k_fold_cross_validation(train_kernel_model, X, y, k=k, epochs=epochs, is_nn=False)
    else:
        raise ValueError("Invalid model choice")
    print(f"Average MSE: {np.mean(mse_scores):.4f}, Average MAE: {np.mean(mae_scores):.4f}")

def spectral_unmixing(S, a):
    return np.dot(S, a)

def plot_spectral_signatures(wavelengths, spectra, title="Spectral Signatures"):
    plt.figure()
    for spec in spectra:
        plt.plot(wavelengths, spec)
    plt.title(title)
    plt.xlabel("Wavelength")
    plt.ylabel("Reflectance")
    plt.show()

def log_training_details(details, filename="training_log.txt"):
    with open(filename, "a") as f:
        f.write(details + "\n")

def save_model_checkpoint(model, filename="model_checkpoint.h5"):
    model.save(filename)

def main():
    file_path = "data.csv"
    try:
        df, reflectance_cols, abundance_cols = load_and_preprocess_data(file_path)
    except Exception as e:
        print(f"Error loading data from '{file_path}': {e}")
        print("Generating dummy data for demonstration.")
        num_samples = 100
        num_wavelengths = 10
        num_endmembers = 3
        data = {"spectrum_id": np.arange(num_samples)}
        for i in range(num_wavelengths):
            data[f"wl_{i}"] = np.random.rand(num_samples)
        for j in range(num_endmembers):
            data[f"abundance_{j}"] = np.random.rand(num_samples)
        df = pd.DataFrame(data)
        reflectance_cols = [col for col in df.columns if col.startswith("wl_")]
        abundance_cols = [col for col in df.columns if col.startswith("abundance_")]
        scaler_ref = MinMaxScaler()
        df[reflectance_cols] = scaler_ref.fit_transform(df[reflectance_cols])
        scaler_ab = MinMaxScaler()
        df[abundance_cols] = scaler_ab.fit_transform(df[abundance_cols])
    augmented_df = data_augmentation(df, reflectance_cols)
    X = augmented_df[reflectance_cols].values
    y = augmented_df[abundance_cols].values
    train_and_evaluate(X, y, model_choice='baseline', k=5, epochs=50)
    train_and_evaluate(X, y, model_choice='nn', k=5, epochs=50)
    train_and_evaluate(X, y, model_choice='kernel', k=5, epochs=50)
    S = np.random.rand(len(reflectance_cols), len(abundance_cols))
    a = y[0]
    reconstructed_spectrum = spectral_unmixing(S, a)
    wavelengths = [int(col.split("_")[1]) for col in reflectance_cols]
    plot_spectral_signatures(wavelengths, [reconstructed_spectrum], title="Reconstructed Spectrum")
    log_training_details("Training completed successfully.")
    nn_model = build_nn_model(X.shape[1], y.shape[1])
    nn_model.fit(X, y, epochs=10, verbose=0)
    save_model_checkpoint(nn_model, "nn_model_checkpoint.h5")
    print("Neural network model checkpoint saved as 'nn_model_checkpoint.h5'.")

if __name__ == "__main__":
    main()
