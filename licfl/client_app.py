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
        self.model.set_weights(parameters)

        # Inference time
        start_infer = time.time()
        y_pred_probs = self.model.predict(self.x_test, verbose=0)
        end_infer = time.time()

        # Convert to class labels (assuming binary classification or categorical softmax)
        if y_pred_probs.shape[1] == 1:
            y_pred = (y_pred_probs > 0.5).astype(int)
        else:
            y_pred = np.argmax(y_pred_probs, axis=1)

        y_true = self.y_test

        # Accuracy from model.evaluate (optional but kept for parity)
        start_train = time.time()
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        end_train = time.time()

        # Basic Metrics
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        auc = roc_auc_score(y_true, y_pred_probs, multi_class="ovr") if y_pred_probs.shape[1] > 1 else roc_auc_score(y_true, y_pred_probs)
        
        # Error Metrics
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # Model efficiency metrics
        param_count = self.model.count_params()
        flops = 0  # Set to 0 or compute if you have a FLOP profiler
        inference_time = end_infer - start_infer
        training_time = end_train - start_train

        return loss, len(self.x_test), {
            "accuracy": accuracy,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "mae": mae,
            "mape": mape,
            "mse": mse,
            "rmse": rmse,
            "parameters": param_count,
            "flops": flops,
            "inference_time": inference_time,
            "training_time": training_time,
        }



def client_fn(context: Context):
    # Load model and data
    net = load_model()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    # Return Client instance
    return FlowerClient(
        net, data, epochs, batch_size, verbose
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
