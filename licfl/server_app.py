"""LICFL: A Flower / TensorFlow app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from licfl.task import load_model


def aggregate_metrics(metrics):
    total_examples = 0
    summed_metrics = {
        "accuracy": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0,
        "parameters": 0.0, "flops": 0.0, "inference_time": 0.0, "training_time": 0.0,
        "mae": 0.0, "mape": 0.0, "rmse": 0.0, "mse": 0.0,
    }

    for num_examples, metric_dict in metrics:
        total_examples += num_examples
        for k in summed_metrics.keys():
            if k in metric_dict:
                summed_metrics[k] += num_examples * metric_dict[k]

    averaged_metrics = {k: v / total_examples for k, v in summed_metrics.items()}
    return averaged_metrics

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # Define strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
