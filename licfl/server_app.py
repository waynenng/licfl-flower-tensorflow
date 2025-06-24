"""LICFL: A Flower / TensorFlow app."""

from sklearn.cluster import KMeans
from numpy.linalg import eig

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from licfl.task import load_model

from typing import List, Tuple, Dict, Any
import flwr as fl
import numpy as np
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitRes,
    Parameters,
)
from flwr.server.strategy import FedAvg, FedAdam, FedAdagrad, FedYogi
from sklearn.cluster import KMeans
from numpy.linalg import eig

from flwr.common import EvaluateIns

class LICFLStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        model,
        num_cohorts: int,
        threshold_improvement: float = 0.01,
        fraction_fit: float = 1.0,
        fraction_eval: float = 1.0,
        min_available_clients: int = 2,
    ):
        # Base model + cohorts config
        self.model = model
        self.num_cohorts = num_cohorts
        self.threshold_improvement = threshold_improvement
        self.previous_loss = float("inf")

        # Initial parameters
        self.initial_parameters = ndarrays_to_parameters(model.get_weights())

        # Instantiate each optimizer strategy
        self.strategies: Dict[str, fl.server.strategy.Strategy] = {
            "FedAvg": FedAvg(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_eval,
                min_available_clients=min_available_clients,
                initial_parameters=self.initial_parameters,
            ),
            "FedAdam": FedAdam(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_eval,
                min_available_clients=min_available_clients,
                initial_parameters=self.initial_parameters,
            ),
            "FedAdagrad": FedAdagrad(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_eval,
                min_available_clients=min_available_clients,
                initial_parameters=self.initial_parameters,
            ),
            "FedYogi": FedYogi(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_eval,
                min_available_clients=min_available_clients,
                initial_parameters=self.initial_parameters,
            ),
        }

        # Placeholders filled each round
        self.cohorts: Dict[int, List[int]] = {}
        self.cohort_parameters: Dict[int, List[np.ndarray]] = {}

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, rnd, parameters, client_manager):
        """Broadcast parameters to each cohort (or all clients in round 1)."""
        clients = client_manager.all()
        instructions = []

        if rnd == 1:
            # Round 1: everyone gets global parameters
            for c in clients:
                instructions.append((c, fl.common.FitIns(parameters, {})))
        else:
            # Subsequent rounds: each cohort gets its own parameters
            for cohort_idx, client_indices in self.cohorts.items():
                cohort_params = ndarrays_to_parameters(self.cohort_parameters[cohort_idx])
                for idx in client_indices:
                    c = clients[idx]
                    instructions.append((c, fl.common.FitIns(cohort_params, {})))
        return instructions

    def aggregate_fit(
        self, rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Parameters, Dict[str, Any]]:
        if not results:
            return None, {}

        # 1) Extract raw ndarrays and losses
        client_ndarrays = [parameters_to_ndarrays(res.parameters) for _, res in results]
        losses = [res.metrics.get("loss", 0.0) for _, res in results]
        current_loss = float(np.mean(losses))

        # 2) Choose optimizer
        strategy_name = self.adaptive_strategy(rnd, self.previous_loss, current_loss)
        optimizer = self.strategies[strategy_name]

        # 3) Cohort formation (round 1 uses Algorithm 2; else reuse)
        if rnd == 1:
            self.cohorts = self.cohorting_algorithm(
                client_ndarrays, n_components=5, n_clusters=self.num_cohorts
            )

        # 4) Per-cohort aggregation via chosen optimizer
        new_cohort_params = {}
        for cohort_idx, client_indices in self.cohorts.items():
            cohort_results = [results[i] for i in client_indices]
            # Delegate to underlying strategy’s aggregate_fit
            params_cohort, _ = optimizer.aggregate_fit(rnd, cohort_results, failures)
            new_cohort_params[cohort_idx] = parameters_to_ndarrays(params_cohort)

        self.cohort_parameters = new_cohort_params

        # 5) Global aggregation across cohorts
        # Build fake FitRes list from cohort aggregates
        fake_results = []
        for cohort_idx, nds in new_cohort_params.items():
            fake_parameters = ndarrays_to_parameters(nds)
            fake_results.append(
                (None, FitRes(parameters=fake_parameters, num_examples=1, metrics={}))
            )
        global_params, metrics = optimizer.aggregate_fit(rnd, fake_results, failures)

        # 6) Update for next round
        self.previous_loss = current_loss

        return global_params, {**metrics, "strategy": strategy_name}

    def cohorting_algorithm(self, client_parameters: List[List[np.ndarray]], n_components: int, n_clusters: int):
        # Step 1: Flatten parameters for each client
        flattened_params = [np.concatenate([layer.flatten() for layer in client_param]) for client_param in client_parameters]

        # Stack all client parameters into matrix X (K x ParamSize)
        X = np.vstack(flattened_params)

        # Step 2: Eigen decomposition for dimensionality reduction (Principal Components)
        covariance_matrix = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = eig(covariance_matrix)
        
        # Select top-n principal components
        top_indices = np.argsort(eigenvalues)[::-1][:n_components]
        principal_components = eigenvectors[:, top_indices]

        # Project X onto principal components
        Y = X @ principal_components

        # Step 3: Build similarity matrix A
        sigma = np.std(Y)
        similarity_matrix = np.exp(-np.square(np.linalg.norm(Y[:, None] - Y[None, :], axis=-1)) / (2 * sigma ** 2))
        np.fill_diagonal(similarity_matrix, 0)  # Set diagonal to 0
        
        # Step 4: Spectral Clustering
        D = np.diag(np.sum(similarity_matrix, axis=1))
        L = np.linalg.inv(np.sqrt(D)) @ similarity_matrix @ np.linalg.inv(np.sqrt(D))

        # Eigen decomposition of Laplacian
        eigvals_L, eigvecs_L = eig(L)
        top_indices_L = np.argsort(eigvals_L)[::-1][:n_clusters]
        embedding = eigvecs_L[:, top_indices_L]

        # Normalize rows
        embedding_normalized = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

        # Step 5: Cluster clients using KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embedding_normalized)
        labels = kmeans.labels_

        # Organize clients into cohorts
        cohorts = {}
        for idx, label in enumerate(labels):
            if label not in cohorts:
                cohorts[label] = []
            cohorts[label].append(idx)

        return cohorts

    @staticmethod
    def aggregate_parameters(params_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Simple FedAvg aggregation"""
        return [np.mean(layer, axis=0) for layer in zip(*params_list)]

    def create_initial_cohorts(self, client_parameters: List[List[np.ndarray]]) -> Dict[int, List[int]]:
        """Placeholder function. Later replaced by Algorithm 2."""
        num_clients = len(client_parameters)
        cohort_size = num_clients // self.num_cohorts
        cohorts = {
            i: list(range(i * cohort_size, (i + 1) * cohort_size))
            for i in range(self.num_cohorts)
        }
        return cohorts

    def adaptive_strategy(self, rnd: int, previous_loss: float, current_loss: float):
        """Dynamically select federated optimization method."""
        delta = previous_loss - current_loss

        if delta > self.threshold_improvement:
            selected_strategy = "FedAvg"
        elif 0 < delta <= self.threshold_improvement:
            selected_strategy = "FedAdam"
        elif delta == 0:
            selected_strategy = "FedAdagrad"
        else:
            selected_strategy = "FedYogi"

        return selected_strategy

    def configure_evaluate(self, rnd, parameters, client_manager):
        """Broadcast global parameters for client‐side evaluation."""
        eval_ins = EvaluateIns(parameters, {})
        return [(client, eval_ins) for client in client_manager.all()]

    def aggregate_evaluate(self, rnd, results, failures):
        """Aggregate client evaluation losses into a single server metric."""
        # results: List of (ClientProxy, EvaluateRes) tuples
        if not results:
            return None, {}
        total_examples = sum(res.num_examples for _, res in results)
        if total_examples == 0:
            return None, {}
        weighted_loss = sum(res.loss * res.num_examples for _, res in results) / total_examples
        return weighted_loss, {}

    def evaluate(self, rnd, parameters, config):
      """No server-side (held-out) evaluation configured."""
      return None, {}

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

from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from licfl.task import load_model


def server_fn(context):
    num_rounds = context.run_config["num-server-rounds"]
    model = load_model(window_size=96, num_features=1)

    strategy = LICFLStrategy(model=model, num_cohorts=3)

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
