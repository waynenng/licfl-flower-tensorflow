import os
import traceback
import pathlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dropout,
    Dense
)

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Global caches to avoid re-loading
# data_cache: list of (X_tr, y_tr, X_te, y_te) per region
# region_cols_list: names of regions corresponding to data_cache indices
data_cache = None
region_cols_list = None

WINDOW_SIZE = 96  # last 96 timesteps per sample

# Columns containing date/time information to drop
DATE_TIME_KEYWORDS = ("date", "time")

# Cache for federated partitions
data_cache = None  # type: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def load_model(window_size: int = WINDOW_SIZE, num_features: int = 1) -> Sequential:
    model = Sequential()

    # 1) Temporal (LSTM) stack
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

    # 2) Spatial pattern extraction with 1D convolutions
    #    Conv along the time axis on the 100-dimensional features
    model.add(Conv1D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(32, kernel_size=3, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # 3) Global pooling to reduce parameters
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.2))

    # 4) Dense head with light regularization
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="linear"))

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def load_data(partition_id: int, num_partitions: int):
    """Load, partition by region, and cache the ERCOT dataset for federated clients."""
    global data_cache, region_cols_list

    if data_cache is None:
        # 1) Locate the Feather file, with env override
        pkg_data = pathlib.Path(__file__).parent / "data" / "ercot-2021-load_profiles.feather"
        proj_data = pathlib.Path.cwd() / "data" / "ercot-2021-load_profiles.feather"
        root_file = pathlib.Path(__file__).parent.parent / "ercot-2021-load_profiles.feather"

        env_path = os.getenv("ERCOT_DATA_PATH")
        candidates = [pathlib.Path(env_path)] if env_path else [pkg_data, proj_data, root_file]

        data_path = None
        for candidate in candidates:
            if candidate and candidate.exists():
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
            traceback.print_exc()
            raise

        # 3) Identify region columns: numeric and not date/time
        region_cols = [
            col for col in df.columns
            if np.issubdtype(df[col].dtype, np.number)
            and not any(kw in col.lower() for kw in DATE_TIME_KEYWORDS)
        ]
        if num_partitions != len(region_cols):
            raise ValueError(
                f"num_partitions ({num_partitions}) must equal number of regions ({len(region_cols)})"
            )

        # Initialize caches
        region_cols_list = region_cols
        data_cache = []

        # 4) Build sliding windows and time-split per region
        for region in region_cols_list:
            series = df[region].values
            X_windows, y_windows = [], []
            for i in range(len(series) - WINDOW_SIZE):
                X_windows.append(series[i:i + WINDOW_SIZE])
                y_windows.append(series[i + WINDOW_SIZE])
            X_arr = np.array(X_windows)[..., np.newaxis]
            y_arr = np.array(y_windows)

            split_idx = int(len(X_arr) * 0.8)
            X_tr, X_te = X_arr[:split_idx], X_arr[split_idx:]
            y_tr, y_te = y_arr[:split_idx], y_arr[split_idx:]

            data_cache.append((X_tr, y_tr, X_te, y_te))

    # Retrieve partition data and region name
    X_train, y_train, X_test, y_test = data_cache[partition_id]
    region_name = region_cols_list[partition_id]
    print(
        f"[load_data] Client {partition_id+1}/{num_partitions} "
        f"(region: {region_name}) → train={X_train.shape}, test={X_test.shape}"
    )
    return X_train, y_train, X_test, y_test