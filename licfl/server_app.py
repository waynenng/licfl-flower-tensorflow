from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedYogi
from licfl.task import load_model
from math import ceil
from typing import List, Tuple
import numpy as np
from flwr.common import Metrics
from flwr.common import EvaluateRes
from sklearn.metrics import mean_absolute_error, mean_squared_error


class FedYogiWithMetrics(FedYogi):
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[int, EvaluateRes]],
        failures: List
    ) -> Tuple[float, dict]:

        # Default loss aggregation (weighted average)
        loss_aggregated, _ = super().aggregate_evaluate(rnd, results, failures)

        # Collect all metrics
        maes, rmses, mses = [], [], []

        for _, eval_res in results:
            metrics = eval_res.metrics
            if metrics:
                maes.append(metrics.get("mae"))
                rmses.append(metrics.get("rmse"))
                mses.append(metrics.get("mse"))

        # Compute average of each metric if available
        metrics_aggregated = {}
        if maes:  metrics_aggregated["mae"] = float(np.mean(maes))
        if rmses: metrics_aggregated["rmse"] = float(np.mean(rmses))
        if mses:   metrics_aggregated["mse"]  = float(np.mean(mses))

        return loss_aggregated, metrics_aggregated

num_regions = 248

def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize global model
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # Define strategy
    strategy = FedYogiWithMetrics(
        # Client Sampling
        fraction_fit=0.15,                                
        min_fit_clients=max(2, ceil(0.15 * num_regions)), 
        fraction_evaluate=0.05,                           
        min_evaluate_clients=max(2, ceil(0.05 * num_regions)),  

        # Availability 
        min_available_clients=num_regions,                

        # Learning Rates & Adaptivity 
        eta=1e-3,           
        eta_l=1e-2,         
        beta_1=0.9,         
        beta_2=0.99,        
        tau=1e-3,           

        # Dynamic Round Configs 
        on_fit_config_fn=lambda rnd: {"local_epochs": 1 if rnd < 10 else 5},
        on_evaluate_config_fn=lambda rnd: {"val_steps": 10 if rnd % 5 == 0 else 2},

        # Starting Point 
        initial_parameters=parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
