"""LICFL: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from licfl.task import load_data, load_model

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


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, batch_size, verbose
    ):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        # Load weights
        self.model.set_weights(parameters)

        # Direct regression eval: returns [mse, mae] since you compiled with loss="mse", metrics=["mae"]
        loss, mae = self.model.evaluate(self.x_test, self.y_test, verbose=0)

        # Return loss for strategy, plus mae as a metric
        return loss, len(self.x_test), {"loss": loss, "mae": mae}



def client_fn(context: Context):
    # 1) Load model, but print full traceback on error
    try:
        net = load_model()
    except Exception:
        import traceback
        traceback.print_exc()
        # re‐raise so Flower still knows this client failed
        raise

    # 2) Determine partitioning params
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 1)

    # 3) Load data, but print full traceback on error
    try:
        data = load_data(partition_id, num_partitions)
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
    # When run as a script, start the client
    app.run()
