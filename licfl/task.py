"""LICFL: A Flower / TensorFlow app."""

import os

import keras
from keras import layers

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

WINDOW_SIZE = 96  # e.g. last 96 timesteps per sample

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model():
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
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
