from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAdagrad
from licfl.task import load_model
from typing import List, Tuple
import numpy as np
from flwr.common import EvaluateRes

class FedAdagradWithMetrics(FedAdagrad):
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[int, EvaluateRes]],
        failures: List
    ) -> Tuple[float, dict]:

        # Default loss aggregation (weighted average)
        loss_aggregated, _ = super().aggregate_evaluate(rnd, results, failures)

        # Collect all metrics
        maes, rmses, mses, mapes = [], [], [], []

        for _, eval_res in results:
            metrics = eval_res.metrics
            if metrics:
                maes.append(metrics.get("mae"))
                rmses.append(metrics.get("rmse"))
                mses.append(metrics.get("mse"))
                mapes.append(metrics.get("mape"))

        # Compute average of each metric if available
        metrics_aggregated = {}
        if maes: metrics_aggregated["mae"] = np.mean(maes)
        if rmses: metrics_aggregated["rmse"] = np.mean(rmses)
        if mses: metrics_aggregated["mse"]  = np.mean(mses)
        if mapes: metrics_aggregated["mape"] = np.mean(mapes)

        return loss_aggregated, metrics_aggregated

num_regions = 50

def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize global model
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # Define strategy
    strategy = FedAdagradWithMetrics(

        # Client sampling
        fraction_fit=1.0,                                
        min_fit_clients=num_regions, 
        fraction_evaluate=1.0,
        min_evaluate_clients=num_regions,  

        # Availability 
        min_available_clients=num_regions,                

        # Learning rates and adaptivity 
        eta=1e-5,           
        eta_l=5e-5,       
        tau=1e-1,           

        # Starting point 
        initial_parameters=parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
