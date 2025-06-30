"""LICFL: A Flower / TensorFlow app."""

import inspect
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from licfl.task import load_data, load_model

# Debug: confirm we’re pulling in the right functions
print(f"[client_app] load_data  from: {inspect.getsourcefile(load_data)}")
print(f"[client_app] load_model from: {inspect.getsourcefile(load_model)}")

import time
import numpy as np
from sklearn.metrics import (
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

from typing import Dict, Any, List, Tuple
import numpy as np
import flwr
from flwr.common import parameters_to_ndarrays  # for real runs

# Define Flower Client and client_fn
class FlowerClient(flwr.client.NumPyClient):
    def __init__(self, model, data, epochs, batch_size, verbose):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        # Always return a plain list of ndarrays
        return self.model.get_weights()

    def fit(
        self, 
        parameters: Any, 
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        # 1) Convert if Flower passed a Parameters proto
        if not isinstance(parameters, list):
            parameters = parameters_to_ndarrays(parameters)

        # 2) (Optional) debug shapes
        print(">>> fit: incoming shapes:", [p.shape for p in parameters])
        print(">>> fit: model needs shapes:", [w.shape for w in self.model.get_weights()])

        # 3) Load the global weights into our local model
        self.model.set_weights(parameters)

        # 4) Train
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=config.get("local_epochs", self.epochs),
            batch_size=config.get("batch_size", self.batch_size),
            verbose=self.verbose,
        )

        # 5) Return updated weights + num samples
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(
        self, 
        parameters: Any, 
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, float]]:
        # Same conversion dance
        if not isinstance(parameters, list):
            parameters = parameters_to_ndarrays(parameters)
        self.model.set_weights(parameters)

        loss, accuracy = self.model.evaluate(
            self.x_test, 
            self.y_test, 
            verbose=self.verbose
        )
        return float(loss), len(self.x_test), {"accuracy": float(accuracy)}



def client_fn(context: Context):
    # 1) Load model, with full traceback on error
    try:
        net = load_model()
    except Exception:
        import traceback
        traceback.print_exc()
        raise

    # 2) Determine partitioning params
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 1)

    # 3) Load data with debug prints
    try:
        print(f"[client_fn] ➡ calling load_data(partition_id={partition_id}, num_partitions={num_partitions})")
        data = load_data(partition_id, num_partitions)
        X_tr, y_tr, X_te, y_te = data
        print(f"[client_fn] ✅ shapes: X_tr={X_tr.shape}, y_tr={y_tr.shape}, X_te={X_te.shape}, y_te={y_te.shape}")
    except Exception:
        import traceback
        traceback.print_exc()
        raise

    # 4) Pull hyperparams (with sane defaults)
    epochs     = context.run_config.get("local-epochs", 1)
    batch_size = context.run_config.get("batch-size",   32)
    verbose    = context.run_config.get("verbose",      0)

    # 5) Return the Flower client
    return FlowerClient(
        net, data, epochs, batch_size, verbose
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)

if __name__ == "__main__":
    app.run()
