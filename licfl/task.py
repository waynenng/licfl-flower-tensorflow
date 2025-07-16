import os
import traceback
import pathlib
from tensorflow.keras import Input
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

data_cache = None
region_cols_list = None

WINDOW_SIZE = 96  # last 96 timesteps per sample

# Columns containing date/time information to drop
DATE_TIME_KEYWORDS = ("date", "time")

EXCLUDED_REGIONS = {

    # excluded these 48 regions for being outliers     
    "BUSOGFDG_COAST","BUSOGFDG_EAST","BUSOGFDG_FWEST","BUSOGFDG_NCENT",
    "BUSOGFDG_NORTH","BUSOGFDG_SCENT","BUSOGFDG_SOUTH","BUSOGFDG_WEST",
    "BUSOGFLT_COAST","BUSOGFLT_EAST","BUSOGFLT_FWEST","BUSOGFLT_NCENT",
    "BUSOGFLT_NORTH","BUSOGFLT_SCENT","BUSOGFLT_SOUTH","BUSOGFLT_WEST",
    "BUSOGFPV_COAST","BUSOGFPV_EAST","BUSOGFPV_FWEST","BUSOGFPV_NCENT",
    "BUSOGFPV_NORTH","BUSOGFPV_SCENT","BUSOGFPV_SOUTH","BUSOGFPV_WEST",
    "BUSOGFWD_COAST","BUSOGFWD_EAST","BUSOGFWD_FWEST","BUSOGFWD_NCENT",
    "BUSOGFWD_NORTH","BUSOGFWD_SCENT","BUSOGFWD_SOUTH","BUSOGFWD_WEST",
    "NMFLAT_COAST","NMFLAT_EAST","NMFLAT_FWEST","NMFLAT_NCENT",
    "NMFLAT_NORTH","NMFLAT_SCENT","NMFLAT_SOUTH","NMFLAT_WEST",
    "NMLIGHT_COAST","NMLIGHT_EAST","NMLIGHT_FWEST","NMLIGHT_NCENT",
    "NMLIGHT_NORTH","NMLIGHT_SCENT","NMLIGHT_SOUTH","NMLIGHT_WEST",

    # excluded these 100 regions simply due to machine application memory constraints 
    "BUSHIDG_COAST", "BUSHIDG_EAST", "BUSHIDG_FWEST", "BUSHIDG_NCENT", "BUSHIDG_NORTH",
    "BUSHIDG_SCENT", "BUSHIDG_SOUTH", "BUSHIDG_WEST", "BUSHILF_COAST", "BUSHILF_EAST",
    "BUSHILF_FWEST", "BUSHILF_NCENT", "BUSHILF_NORTH", "BUSHILF_SCENT", "BUSHILF_SOUTH",
    "BUSHILF_WEST", "BUSHIPV_COAST", "BUSHIPV_EAST", "BUSHIPV_FWEST", "BUSHIPV_NCENT",
    "BUSHIPV_NORTH", "BUSHIPV_SCENT", "BUSHIPV_SOUTH", "BUSHIPV_WEST", "BUSHIWD_COAST",
    "BUSHIWD_EAST", "BUSHIWD_FWEST", "BUSHIWD_NCENT", "BUSHIWD_NORTH", "BUSHIWD_SCENT",
    "BUSHIWD_SOUTH", "BUSHIWD_WEST", "BUSIDRRQ_COAST", "BUSIDRRQ_EAST", "BUSIDRRQ_FWEST", 
    "BUSIDRRQ_NCENT", "BUSIDRRQ_NORTH", "BUSIDRRQ_SCENT", "BUSIDRRQ_SOUTH", "BUSIDRRQ_WEST", 
    "BUSLODG_COAST", "BUSLODG_EAST", "BUSLODG_FWEST", "BUSLODG_NCENT", "BUSLODG_NORTH",
    "BUSLODG_SCENT", "BUSLODG_WEST", "BUSLOLF_COAST", "BUSLOLF_EAST", "BUSLOLF_FWEST", 
    "BUSLOLF_NCENT", "BUSLOLF_NORTH", "BUSLOLF_SCENT", "BUSLOLF_WEST", "BUSLOPV_COAST", 
    "BUSLOPV_EAST", "BUSLOPV_FWEST", "BUSLOPV_NCENT", "BUSLOPV_NORTH", "BUSLOPV_SCENT", 
    "BUSLOPV_SOUTH", "BUSLOPV_WEST", "BUSLOWD_COAST", "BUSLOWD_EAST", "BUSLOWD_FWEST",
    "BUSLOWD_NCENT", "BUSLOWD_SCENT", "BUSLOWD_WEST", "BUSMEDDG_COAST", "BUSMEDDG_EAST",
    "BUSMEDDG_FWEST", "BUSMEDDG_NCENT", "BUSMEDDG_NORTH", "BUSMEDDG_SCENT", "BUSMEDDG_SOUTH", 
    "BUSMEDDG_WEST", "BUSMEDLF_COAST", "BUSMEDLF_EAST", "BUSMEDLF_FWEST", "BUSMEDLF_NCENT",
    "BUSMEDLF_NORTH", "BUSMEDLF_SCENT", "BUSMEDLF_SOUTH", "BUSMEDLF_WEST", "BUSMEDPV_COAST", 
    "BUSMEDPV_EAST", "BUSMEDPV_FWEST", "BUSMEDPV_NCENT", "BUSMEDPV_NORTH", "BUSMEDPV_SCENT", 
    "BUSMEDPV_SOUTH", "BUSMEDPV_WEST", "BUSMEDWD_COAST", "BUSMEDWD_EAST", "BUSMEDWD_FWEST", 
    "BUSMEDWD_NCENT", "BUSMEDWD_NORTH", "BUSMEDWD_SCENT", "BUSMEDWD_SOUTH", "BUSMEDWD_WEST",
}

def load_model(window_size: int = WINDOW_SIZE, num_features: int = 1) -> Sequential:
    
    model = Sequential()
    model.add(Input(shape=(window_size, num_features)))

    # 1) Temporal (LSTM) stack
    model.add(
        LSTM(
            100,
            activation="tanh",
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
            col
            for col in df.columns
            if (
                np.issubdtype(df[col].dtype, np.number)
                and not any(kw in col.lower() for kw in DATE_TIME_KEYWORDS)
                and col not in EXCLUDED_REGIONS
            )
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