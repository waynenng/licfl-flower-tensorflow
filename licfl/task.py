import os
import traceback
import pathlib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input
import pandas as pd
import numpy as np
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
region_scalers = None

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

    # below regions must also be considered for exclusion simply due to machine application memory constraints
     
    # "BUSHIDG_COAST", "BUSHIDG_EAST", "BUSHIDG_FWEST", "BUSHIDG_NCENT", "BUSHIDG_NORTH",
    "BUSHIDG_SCENT", "BUSHIDG_SOUTH", "BUSHIDG_WEST", "BUSHILF_COAST", "BUSHILF_EAST",
    "BUSHILF_FWEST", "BUSHILF_NCENT", "BUSHILF_NORTH", "BUSHILF_SCENT", "BUSHILF_SOUTH",
    "BUSHILF_WEST", "BUSHIPV_COAST", "BUSHIPV_EAST", "BUSHIPV_FWEST", "BUSHIPV_NCENT",
    "BUSHIPV_NORTH", "BUSHIPV_SCENT", "BUSHIPV_SOUTH", "BUSHIPV_WEST", "BUSHIWD_COAST",
    "BUSHIWD_EAST", "BUSHIWD_FWEST", "BUSHIWD_NCENT", "BUSHIWD_NORTH", "BUSHIWD_SCENT",
    "BUSHIWD_SOUTH", "BUSHIWD_WEST", "BUSIDRRQ_COAST", "BUSIDRRQ_EAST", "BUSIDRRQ_FWEST", 
    "BUSIDRRQ_NCENT", "BUSIDRRQ_NORTH", "BUSIDRRQ_SCENT", "BUSIDRRQ_SOUTH", "BUSIDRRQ_WEST", 
    # "BUSLODG_COAST", "BUSLODG_EAST", "BUSLODG_FWEST", "BUSLODG_NCENT", "BUSLODG_NORTH",
    "BUSLODG_SCENT", "BUSLODG_WEST", "BUSLOLF_COAST", "BUSLOLF_EAST", "BUSLOLF_FWEST", 
    "BUSLOLF_NCENT", "BUSLOLF_NORTH", "BUSLOLF_SCENT", "BUSLOLF_WEST", "BUSLOPV_COAST", 
    # "BUSLOPV_EAST", "BUSLOPV_FWEST", "BUSLOPV_NCENT", "BUSLOPV_NORTH", "BUSLOPV_SCENT", 
    "BUSLOPV_SOUTH", "BUSLOPV_WEST", "BUSLOWD_COAST", "BUSLOWD_EAST", "BUSLOWD_FWEST",
    "BUSLOWD_NCENT", "BUSLOWD_SCENT", "BUSLOWD_WEST", "BUSMEDDG_COAST", "BUSMEDDG_EAST",
    # "BUSMEDDG_FWEST", "BUSMEDDG_NCENT", "BUSMEDDG_NORTH", "BUSMEDDG_SCENT", "BUSMEDDG_SOUTH", 
    "BUSMEDDG_WEST", "BUSMEDLF_COAST", "BUSMEDLF_EAST", "BUSMEDLF_FWEST", "BUSMEDLF_NCENT",
    "BUSMEDLF_NORTH", "BUSMEDLF_SCENT", "BUSMEDLF_SOUTH", "BUSMEDLF_WEST", "BUSMEDPV_COAST", 
    # "BUSMEDPV_EAST", "BUSMEDPV_FWEST", "BUSMEDPV_NCENT", "BUSMEDPV_NORTH", "BUSMEDPV_SCENT", 
    "BUSMEDPV_SOUTH", "BUSMEDPV_WEST", "BUSMEDWD_COAST", "BUSMEDWD_EAST", "BUSMEDWD_FWEST", 
    "BUSMEDWD_NCENT", "BUSMEDWD_NORTH", "BUSMEDWD_SCENT", "BUSMEDWD_SOUTH", "BUSMEDWD_WEST",

    "BUSLODG_SOUTH", "BUSLOLF_SOUTH", "BUSLOWD_NORTH", "BUSLOWD_SOUTH", "BUSNODDG_COAST",	
    "BUSNODDG_EAST", "BUSNODDG_FWEST", "BUSNODDG_NCENT", "BUSNODDG_NORTH", "BUSNODDG_SCENT",	
    "BUSNODDG_SOUTH", "BUSNODDG_WEST", "BUSNODEM_COAST", "BUSNODEM_EAST", "BUSNODEM_FWEST",	
    "BUSNODEM_NCENT", "BUSNODEM_NORTH",	"BUSNODEM_SCENT", "BUSNODEM_SOUTH", "BUSNODEM_WEST",	
    "BUSNODPV_COAST", "BUSNODPV_EAST", "BUSNODPV_FWEST", "BUSNODPV_NCENT", "BUSNODPV_NORTH",
    "BUSNODPV_SCENT", "BUSNODPV_SOUTH",	"BUSNODPV_WEST", "BUSNODWD_COAST", "BUSNODWD_EAST",
    # "BUSNODWD_FWEST", "BUSNODWD_NCENT", "BUSNODWD_NORTH", "BUSNODWD_SCENT", "BUSNODWD_SOUTH",
    "BUSNODWD_WEST", "RESHIDG_COAST", "RESHIDG_EAST", "RESHIDG_FWEST", "RESHIDG_NCENT",	
    "RESHIDG_NORTH", "RESHIDG_SCENT", "RESHIDG_SOUTH", "RESHIDG_WEST",	"RESHIPV_COAST",
    "RESHIPV_EAST",	"RESHIPV_FWEST", "RESHIPV_NCENT", "RESHIPV_NORTH", "RESHIPV_SCENT",
    "RESHIPV_SOUTH", "RESHIPV_WEST", "RESHIWD_COAST", "RESHIWD_EAST", "RESHIWD_FWEST",
    "RESHIWD_NCENT", "RESHIWD_NORTH", "RESHIWD_SCENT", "RESHIWD_SOUTH",	"RESHIWD_WEST",
    # "RESHIWR_COAST", "RESHIWR_EAST", "RESHIWR_FWEST", "RESHIWR_NCENT", "RESHIWR_NORTH",	
    "RESHIWR_SCENT", "RESHIWR_SOUTH", "RESHIWR_WEST", "RESLODG_COAST", "RESLODG_EAST",
    # "RESLODG_FWEST", "RESLODG_NCENT", "RESLODG_NORTH", "RESLODG_SCENT", "RESLODG_SOUTH",
    "RESLODG_WEST",	"RESLOPV_COAST", "RESLOPV_EAST", "RESLOPV_FWEST", "RESLOPV_NCENT",
    "RESLOPV_NORTH", "RESLOPV_SCENT", "RESLOPV_SOUTH", "RESLOPV_WEST", "RESLOWD_COAST",
    # "RESLOWD_EAST", "RESLOWD_FWEST", "RESLOWD_NCENT", "RESLOWD_NORTH", "RESLOWD_SCENT",
    "RESLOWD_SOUTH", "RESLOWD_WEST", "RESLOWR_COAST", "RESLOWR_EAST", "RESLOWR_FWEST",	
    # "RESLOWR_NCENT", "RESLOWR_NORTH", "RESLOWR_SCENT", "RESLOWR_SOUTH", "RESLOWR_WEST",
}

def load_model(window_size: int = WINDOW_SIZE, num_features: int = 1) -> Sequential:
    
    model = Sequential()
    model.add(Input(shape=(window_size, num_features)))

    # Temporal (LSTM) stack
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

    # Spatial pattern extraction with 1D convolutions
    # Conv along the time axis on the 100-dimensional features
    model.add(Conv1D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(32, kernel_size=3, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Global pooling to reduce parameters
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.2))

    # Dense head with light regularization
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="linear"))

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def load_data(partition_id: int, num_partitions: int):
    
    global data_cache, region_cols_list, region_scalers

    if data_cache is None:
        # Locate the Feather file, with env override
        pkg_data  = Path(__file__).parent / "data" / "ercot-2021-load_profiles.feather"
        proj_data = Path.cwd() / "data" / "ercot-2021-load_profiles.feather"
        root_file = Path(__file__).parent.parent / "ercot-2021-load_profiles.feather"

        env_path = os.getenv("ERCOT_DATA_PATH")
        candidates = [Path(env_path)] if env_path else [pkg_data, proj_data, root_file]

        data_path = None
        for cand in candidates:
            if cand and cand.exists():
                data_path = cand
                break
        if data_path is None:
            raise FileNotFoundError(
                "Could not find 'ercot-2021-load_profiles.feather' in any of: "
                + ", ".join(str(c) for c in candidates)
            )

        # Read into DataFrame
        try:
            df = pd.read_feather(data_path)
        except Exception:
            traceback.print_exc()
            raise

        # Identify region columns: numeric and not date/time
        region_cols = [
            col for col in df.columns
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
        region_scalers = {}

        # Standardize and build windows per region
        for region in region_cols_list:
            raw = df[region].values.reshape(-1, 1)
            split_idx = int(len(raw) * 0.8)
            train_raw = raw[:split_idx]
            test_raw  = raw[split_idx:]

            # Fit scaler on train, transform both
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_raw).flatten()
            test_scaled  = scaler.transform(test_raw).flatten()

            # Build sliding windows on scaled data
            X_tr, y_tr = [], []
            for i in range(len(train_scaled) - WINDOW_SIZE):
                X_tr.append(train_scaled[i : i + WINDOW_SIZE])
                y_tr.append(train_scaled[i + WINDOW_SIZE])
            X_te, y_te = [], []
            for i in range(len(test_scaled) - WINDOW_SIZE):
                X_te.append(test_scaled[i : i + WINDOW_SIZE])
                y_te.append(test_scaled[i + WINDOW_SIZE])

            X_tr = np.array(X_tr)[..., np.newaxis]
            y_tr = np.array(y_tr)
            X_te = np.array(X_te)[..., np.newaxis]
            y_te = np.array(y_te)

            # Cache windows and scaler
            data_cache.append((X_tr, y_tr, X_te, y_te, scaler))
            region_scalers[region] = scaler

    # Retrieve requested partition
    X_train, y_train, X_test, y_test, scaler = data_cache[partition_id]
    region = region_cols_list[partition_id]
    print(
        f"[load_data] Client {partition_id+1}/{num_partitions} "
        f"(region: {region}) | train={X_train.shape}, test={X_test.shape}"
    )
    return X_train, y_train, X_test, y_test, scaler