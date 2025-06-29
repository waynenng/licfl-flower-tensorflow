'''LICFL: A Flower / TensorFlow app.'''  
import os
import traceback
import pathlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Reshape,
    Conv2D,
    BatchNormalization,
    Flatten,
    Dense,
)

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

WINDOW_SIZE = 96  # last 96 timesteps per sample

# Cache for federated partitions
data_cache = None  # type: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def load_model(window_size: int = WINDOW_SIZE, num_features: int = 1) -> Sequential:
    """Build and compile the LSTM-CNN regression model."""
    model = Sequential()

    # ─── Temporal (LSTM) with full sequence output ─────────────────
    model.add(
        LSTM(
            100,
            activation="tanh",
            input_shape=(window_size, num_features),
            return_sequences=True,
        )
    )
    model.add(
        LSTM(
            100,
            activation="tanh",
            return_sequences=True,
        )
    )

    # ─── Spatial (CNN) on LSTM output ─────────────────────────────
    model.add(Reshape((window_size, 100, 1)))
    model.add(Conv2D(24, kernel_size=(4, 1), activation="relu"))
    model.add(Conv2D(36, kernel_size=(11, 1), activation="relu"))
    model.add(Conv2D(48, kernel_size=(3, 1), activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 1), activation="relu"))
    model.add(BatchNormalization())
    model.add(Flatten())

    # ─── Dense Head ──────────────────────────────────────────────
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="linear"))

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def load_data(partition_id: int, num_partitions: int):
    """Load, partition, and cache the ERCOT dataset for federated clients."""
    global data_cache
    if data_cache is None:
        # 1) Locate the Feather file, with env override
        pkg_data = pathlib.Path(__file__).parent / "data" / "ercot-2021-load_profiles.feather"
        proj_data = pathlib.Path.cwd() / "data" / "ercot-2021-load_profiles.feather"
        root_file = pathlib.Path(__file__).parent.parent / "ercot-2021-load_profiles.feather"

        env_path = os.getenv("ERCOT_DATA_PATH")
        candidates = [pathlib.Path(env_path)] if env_path else [pkg_data, proj_data, root_file]

        data_path = None
        for candidate in candidates:
            print(f"[load_data] Checking {candidate} → exists? {candidate.exists()}")
            if candidate.exists():
                data_path = candidate
                break
        if data_path is None:
            raise FileNotFoundError(
                "Could not find 'ercot-2021-load_profiles.feather' in any of: "
                + ", ".join(str(c) for c in candidates)
            )

        # 2) Read into DataFrame
        try:
            df = pd.read_feather(data_path)
        except Exception:
            print(f"[load_data] Failed to read {data_path}:")
            traceback.print_exc()
            raise

        # 3) Aggregate numeric columns (drop datetime)
        series = df.select_dtypes(include=[np.number]).mean(axis=1).values

        # 4) Build sliding windows
        X, y = [], []
        for i in range(len(series) - WINDOW_SIZE):
            X.append(series[i : i + WINDOW_SIZE])
            y.append(series[i + WINDOW_SIZE])
        X = np.array(X)[..., np.newaxis]
        y = np.array(y)

        # 5) IID partition
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        parts = np.array_split(idx, num_partitions)

        # 6) Train/Test for each partition
        data_cache = []
        for part in parts:
            X_c = X[part]
            y_c = y[part]
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_c, y_c, test_size=0.2, random_state=42
            )
            data_cache.append((X_tr, y_tr, X_te, y_te))

    # Return the requested partition
    X_train, y_train, X_test, y_test = data_cache[partition_id]
    print(
        f"[load_data] Client {partition_id+1}/{num_partitions}: "
        f"train={X_train.shape}, test={X_test.shape}"
    )
    return X_train, y_train, X_test, y_test