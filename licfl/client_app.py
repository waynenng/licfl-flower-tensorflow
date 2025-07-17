from flwr.client import ClientApp
from flwr.common import Context
from licfl.task import load_data, load_model
from typing import Dict, Any, List, Tuple
import numpy as np
import flwr
from flwr.common import parameters_to_ndarrays 
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)

# Define Flower Client and client_fn
class FlowerClient(flwr.client.NumPyClient):

    def __init__(self, model, data, epochs, batch_size, verbose):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test, self.scaler = data
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
        # Convert if Flower passed a Parameters proto
        if not isinstance(parameters, list):
            parameters = parameters_to_ndarrays(parameters)

        # Load the global weights into our local model
        self.model.set_weights(parameters)

        # Train
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=config.get("local_epochs", self.epochs),
            batch_size=config.get("batch_size", self.batch_size),
            verbose=self.verbose,
        )

        # Return updated weights + num samples
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(
        self,
        parameters: Any,
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, float]]:

        # Convert Flower Parameters proto to ndarray list if needed
        if not isinstance(parameters, list):
            parameters = parameters_to_ndarrays(parameters)

        # Load the global weights into our model
        self.model.set_weights(parameters)

        # Evaluate loss and MAE on the scaled test data
        loss_scaled, mae_scaled = self.model.evaluate(
            self.x_test, self.y_test, verbose=self.verbose
        )

        # Predict on the scaled test inputs
        y_pred_scaled = self.model.predict(self.x_test, verbose=0).flatten()
        y_true_scaled = self.y_test.flatten()

        # Invert back to original MW units
        # scaler was passed in via load_data and stored as self.scaler
        y_pred = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = self.scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

        # Compute metrics on the *unscaled* values
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        # Return the scaled loss (for FL) and MW‐based metrics
        return loss_scaled, len(self.x_test), {
            "mae": mae,
            "rmse": rmse,
            "mse": mse,
            "mape": mape,
        }

def client_fn(context: Context):

    # Determine partitioning params
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 1)

    # Load data (returns X_tr, y_tr, X_te, y_te, scaler)
    X_tr, y_tr, X_te, y_te, scaler = load_data(partition_id, num_partitions)

    # Build the model with matching input dims
    window_size, num_features = X_tr.shape[1], X_tr.shape[2]
    net = load_model(window_size=window_size, num_features=num_features)

    # Pull hyperparams (with sane defaults)
    epochs     = context.run_config.get("local-epochs", 3)
    batch_size = context.run_config.get("batch-size", 64)
    verbose    = context.run_config.get("verbose", 0)

    # Return the Flower client, passing through the scaler
    return FlowerClient(
        net,
        (X_tr, y_tr, X_te, y_te, scaler),
        epochs,
        batch_size,
        verbose
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)

if __name__ == "__main__":
    app.run()
