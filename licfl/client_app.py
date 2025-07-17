from flwr.client import ClientApp
from flwr.common import Context
from licfl.task import load_data, load_model

# Debug: confirm we’re pulling in the right functions
# print(f"[client_app] load_data  from: {inspect.getsourcefile(load_data)}")
# print(f"[client_app] load_model from: {inspect.getsourcefile(load_model)}")

from typing import Dict, Any, List, Tuple
import numpy as np
import flwr
from flwr.common import parameters_to_ndarrays  # for real runs
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)

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

        # 2) Load the global weights into our local model
        self.model.set_weights(parameters)

        # 3) Train
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=config.get("local_epochs", self.epochs),
            batch_size=config.get("batch_size", self.batch_size),
            verbose=self.verbose,
        )

        # 4) Return updated weights + num samples
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(
        self,
        parameters: Any,
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, float]]:

        if not isinstance(parameters, list):
            parameters = parameters_to_ndarrays(parameters)

        self.model.set_weights(parameters)

        # Get loss and MAE from model's internal evaluation
        loss, mae = self.model.evaluate(self.x_test, self.y_test, verbose=self.verbose)

        # Predict and compute MSE, RMSE, and MAPE
        y_pred = self.model.predict(self.x_test, verbose=0).flatten()
        y_true = self.y_test.flatten()

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        return float(loss), len(self.x_test), {
            "mae": mae,
            "rmse": rmse,
            "mse": mse,
            "mape": mape,
        }

def client_fn(context: Context):
    # 1) Determine partitioning params
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 1)

    # 2) Load data first to determine input shape
    X_tr, y_tr, X_te, y_te = load_data(partition_id, num_partitions)

    # 3) Build the model with matching input dims
    window_size, num_features = X_tr.shape[1], X_tr.shape[2]
    net = load_model(window_size=window_size, num_features=num_features)

    # 4) Pull hyperparams (with sane defaults)
    epochs     = context.run_config.get("local-epochs", 3)
    batch_size = context.run_config.get("batch-size", 64)
    verbose    = context.run_config.get("verbose", 0)

    # 5) Return the Flower client
    return FlowerClient(
        net,
        (X_tr, y_tr, X_te, y_te),
        epochs,
        batch_size,
        verbose
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)

if __name__ == "__main__":
    app.run()
