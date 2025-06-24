"""LICFL: A Flower / TensorFlow app."""

import os

import keras
from keras import layers

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    RepeatVector,
    Reshape,
    Conv2D,
    BatchNormalization,
    Flatten,
    Dense,
)

WINDOW_SIZE = 96  # e.g. last 96 timesteps per sample

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model(window_size: int, num_features: int):
    model = Sequential()

    # ─── Temporal (LSTM) with full sequence output ─────────────────────────────────
    # First LSTM returns full sequence for next layer
    model.add(LSTM(
        100,
        activation="tanh",
        input_shape=(window_size, num_features),
        return_sequences=True,
    ))
    # Second LSTM also returns full sequence
    model.add(LSTM(
        100,
        activation="tanh",
        return_sequences=True,
    ))

    # ─── Spatial (CNN) on LSTM output ──────────────────────────────────────────────
    # Reshape (batch, time, features) -> (batch, time, features, 1)
    # Here features dimension is 100 after LSTM layers
    model.add(Reshape((window_size, 100, 1)))

    # conv1: 24 filters, kernel (4×1)
    model.add(Conv2D(24, kernel_size=(4, 1), activation="relu"))
    # conv2: 36 filters, kernel (11×1)
    model.add(Conv2D(36, kernel_size=(11, 1), activation="relu"))
    # conv3: 48 filters, kernel (3×1) + BatchNorm
    model.add(Conv2D(48, kernel_size=(3, 1), activation="relu"))
    model.add(BatchNormalization())
    # conv4: 32 filters, kernel (3×1) + BatchNorm
    model.add(Conv2D(32, kernel_size=(3, 1), activation="relu"))
    model.add(BatchNormalization())

    model.add(Flatten())

    # ─── Dense Head ────────────────────────────────────────────────────────────────
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="linear"))

    # ─── Compile for regression ────────────────────────────────────────────────────
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    # 1) Read the full ERCOT load time series
    df = pd.read_feather("data/ercot-2021-load_profiles.feather")
    # Suppose df has columns: 'datetime', zone_0, zone_1, ..., zone_N
    # You could pick one zone or aggregate all zones:
    data = df.mean(axis=1).values

    # 2) Convert to sliding windows
    X = []
    y = []
    for i in range(len(data) - WINDOW_SIZE):
        X.append(data[i : i + WINDOW_SIZE])
        y.append(data[i + WINDOW_SIZE])
    X = np.array(X)[..., np.newaxis]  # shape (num_samples, WINDOW_SIZE, 1)
    y = np.array(y)

    # 3) Partition samples across clients
    #    Simple IID split by index:
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    parts = np.array_split(indices, num_partitions)
    client_idx = parts[partition_id]

    X_client = X[client_idx]
    y_client = y[client_idx]

    # 4) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_client, y_client, test_size=0.2, random_state=42
    )

    return X_train, y_train, X_test, y_test
