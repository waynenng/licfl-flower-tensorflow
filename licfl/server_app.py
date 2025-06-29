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
from flwr.server.strategy import FedAvg
from flwr.server.strategy import FedAdam, FedAdagrad, FedYogi
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
        local_epochs: int = 1,      
        batch_size: int = 32,
    ):
        # Base model + cohorts config
        self.model = model
        self.num_cohorts = num_cohorts
        self.threshold_improvement = threshold_improvement
        self.previous_loss = float("inf")
        self.local_epochs = local_epochs    # ADD THIS
        self.batch_size = batch_size

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

        # ─── Algorithm 3 hyperparameters & state ────────────────────────────────────
        # β₁, β₂ for momentum/variance updates, τ for numerical stability, η for step‐size
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.tau = 1e-9
        self.eta = 1.0

        # Per‐cohort Algorithm 3 state (populated in round 1)
        #   theta_prev[j]: last selected flat vector for cohort j
        #   m_prev[j]:      momentum accumulator for cohort j
        #   v_prev[j]:      dict of per‐strategy variance accumulators for cohort j
        self.theta_prev: Dict[int, np.ndarray] = {}
        self.m_prev: Dict[int, np.ndarray] = {}
        self.v_prev: Dict[int, Dict[str, np.ndarray]] = {}

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager, **kwargs):
        """Broadcast parameters to each cohort (or all clients in round 1)."""

        clients = client_manager.all()

        if server_round == 1:
            # ── Round 1, Phase 1: everyone gets the GLOBAL parameters ──
            return [
                (c, fl.common.FitIns(parameters, {}))
                for c in clients
            ]

        # ── Round 2 and beyond (including Round 1, Phase 2 run inside aggregate_fit) ──
        instructions = []
        for cohort_idx, client_indices in self.cohorts.items():
            # Convert your stored numpy arrays back into Flower Parameters
            cohort_params = ndarrays_to_parameters(
                self.cohort_parameters[cohort_idx]
            )
            for idx in client_indices:
                instructions.append(
                    (clients[idx], fl.common.FitIns(cohort_params, {}))
                )
        return instructions

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Parameters, Dict[str, Any]]:
        if not results:
            return None, {}

        # 1) Extract raw ndarrays and compute loss
        client_ndarrays = [parameters_to_ndarrays(res.parameters) for _, res in results]
        losses = [res.metrics.get("loss", 0.0) for _, res in results]
        current_loss = float(np.mean(losses))

        # ───── Round 1: Two-Phase Update with Algorithm 3 ─────
        if rnd == 1:
            # Phase 1: clients update on GLOBAL Θ (build V)
            self.V = client_ndarrays

            # Aggregate to get initial global model (can use FedAvg as a warm start)
            fake_phase1 = [(
                None,
                FitRes(
                    parameters=ndarrays_to_parameters(w),
                    num_examples=1,
                    metrics={},
                ),
            ) for w in self.V]
            global_params, _ = self.strategies["FedAvg"].aggregate_fit(rnd, fake_phase1, failures)
            global_nds = parameters_to_ndarrays(global_params)

            # Cohort assignment
            self.cohorts = self.cohorting_algorithm(
                self.V,
                n_components=5,
                n_clusters=self.num_cohorts,
            )

            # Initialize Algorithm 3 state per cohort
            flat_global = np.concatenate([w.flatten() for w in global_nds])
            for j in self.cohorts:
                self.theta_prev[j] = flat_global.copy()
                self.m_prev[j] = np.zeros_like(flat_global)
                self.v_prev[j] = {
                    "FedAvg":     np.zeros_like(flat_global),
                    "FedAdagrad": np.zeros_like(flat_global),
                    "FedYogi":    np.zeros_like(flat_global),
                    "FedAdam":    np.zeros_like(flat_global),
                }
            self.cohort_parameters = {j: global_nds for j in self.cohorts}

            # Phase 2: re-fit clients on their cohort models
            fit_ins = []
            for j, client_indices in self.cohorts.items():
                ins_params = ndarrays_to_parameters(self.cohort_parameters[j])
                for idx in client_indices:
                    client_proxy, _ = results[idx]
                    fit_ins.append(
                        (
                            client_proxy,
                            FitIns(
                                parameters=ins_params,
                                config={
                                    "epochs": self.local_epochs,
                                    "batch_size": self.batch_size,
                                },
                            ),
                        )
                    )
            results2 = self.client_manager.fit(fit_ins)

            # Per-cohort aggregation via Algorithm 3
            new_cohort_params = {}
            shapes = [w.shape for w in self.cohort_parameters[next(iter(self.cohorts))]]
            splits = np.cumsum([np.prod(s) for s in shapes])[:-1]
            for j, client_indices in self.cohorts.items():
                # collect flat updates
                cohort_updates = []
                for cid, res in results2:
                    if cid in [results[i][0] for i in client_indices]:
                        nds = parameters_to_ndarrays(res.parameters)
                        cohort_updates.append(np.concatenate([l.flatten() for l in nds]))
                # Algorithm 3 selection
                flat_new = self._algorithm3_aggregate(cohort_updates, j)
                # un-flatten
                layers = np.split(flat_new, splits)
                new_cohort_params[j] = [l.reshape(s) for l, s in zip(layers, shapes)]

            self.cohort_parameters = new_cohort_params

            # Global aggregation across cohorts (warm start)
            fake_phase2 = [(
                None,
                FitRes(
                    parameters=ndarrays_to_parameters(w),
                    num_examples=1,
                    metrics={},
                ),
            ) for w in new_cohort_params.values()]
            global_params_final, metrics = self.strategies["FedAvg"].aggregate_fit(
                rnd, fake_phase2, failures
            )

            # Save state and return
            self.previous_loss = current_loss
            metrics["strategy"] = "Algorithm3"
            return global_params_final, metrics

        # ───── Rounds ≥ 2: Algorithm 3 per-cohort aggregation ─────
        new_cohort_params = {}
        shapes = [w.shape for w in self.cohort_parameters[next(iter(self.cohorts))]]
        splits = np.cumsum([np.prod(s) for s in shapes])[:-1]
        for j, client_indices in self.cohorts.items():
            # flatten client updates
            flat_updates = []
            for i in client_indices:
                _, res = results[i]
                nds = parameters_to_ndarrays(res.parameters)
                flat_updates.append(np.concatenate([l.flatten() for l in nds]))
            # Algorithm 3 selection
            flat_new = self._algorithm3_aggregate(flat_updates, j)
            # un-flatten
            layers = np.split(flat_new, splits)
            new_cohort_params[j] = [l.reshape(s) for l, s in zip(layers, shapes)]

        self.cohort_parameters = new_cohort_params

        # Global aggregation across cohorts
        fake_results = [(
            None,
            FitRes(
                parameters=ndarrays_to_parameters(nds),
                num_examples=1,
                metrics={},
            ),
        ) for nds in new_cohort_params.values()]
        global_params, metrics = self.strategies["FedAvg"].aggregate_fit(
            rnd, fake_results, failures
        )

        # Update and return
        self.previous_loss = current_loss
        metrics["strategy"] = "Algorithm3"
        return global_params, metrics

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

    def configure_evaluate(self, server_round, parameters, client_manager, **kwargs):
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

    def evaluate(self, rnd, parameters, config=None):
         """No server-side (held-out) evaluation configured."""
         return None, {}
    
    def _algorithm3_aggregate(
        self, flat_updates: List[np.ndarray], cohort_idx: int
    ) -> np.ndarray:
        """
        Implements Algorithm 3 for one cohort:
        - flat_updates: list of client weight-vectors, each flattened.
        - cohort_idx: which cohort we’re updating.
        Returns the selected new flat parameter‐vector.
        """
        prev = self.theta_prev[cohort_idx]
        # Compute average “drift” Δₙ = (1/|C|) Σ (uₖ − prev)
        deltas = [u - prev for u in flat_updates]
        delta_avg = sum(deltas) / len(deltas)

        # 1) m_r = β1 · m_{r-1} + (1−β1) · Δ
        m_r = self.beta1 * self.m_prev[cohort_idx] + (1 - self.beta1) * delta_avg

        # 2) v_r per‐strategy
        v_r = {}
        # FedAvg wipes out variance
        v_r["FedAvg"] = np.zeros_like(delta_avg)
        # FedAdagrad
        v_r["FedAdagrad"] = (
            self.v_prev[cohort_idx]["FedAdagrad"]
            + np.square(delta_avg)
        )
        # FedYogi
        diff = self.v_prev[cohort_idx]["FedYogi"] - np.square(delta_avg)
        v_r["FedYogi"] = (
            self.v_prev[cohort_idx]["FedYogi"]
            - (1 - self.beta2) * np.square(delta_avg) * np.sign(diff)
        )
        # FedAdam
        v_r["FedAdam"] = (
            self.beta2 * self.v_prev[cohort_idx]["FedAdam"]
            + (1 - self.beta2) * np.square(delta_avg)
        )

        # 3) Compute candidate thetas: Θₙ⁽c⁾ = prev + η · m_r / (√(v_r)+τ)
        candidates: Dict[str, np.ndarray] = {}
        for name, v_vec in v_r.items():
            candidates[name] = prev + self.eta * m_r / (np.sqrt(v_vec) + self.tau)

        # 4) Score by drift:  s_c = ‖Θₙ⁽c⁾‖_F − ‖prev‖_F, pick minimal
        norm_prev = np.linalg.norm(prev)
        scores = {
            name: np.linalg.norm(theta_c) - norm_prev
            for name, theta_c in candidates.items()
        }
        best = min(scores, key=scores.get)

        # 5) Update stored state for next round
        self.m_prev[cohort_idx]       = m_r
        self.v_prev[cohort_idx]       = v_r
        self.theta_prev[cohort_idx]   = candidates[best]

        return candidates[best]

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

    local_epochs = context.run_config.get("local-epochs", 1)
    batch_size = context.run_config.get("batch-size", 32)

    strategy = LICFLStrategy(
        model=model, 
        num_cohorts=3,
        local_epochs=local_epochs,
        batch_size=batch_size
        )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    # When run as a script, start the server
    app.run()
