\
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sklearn.metrics import mean_squared_error


@dataclass
class FitResult:
    best_W: np.ndarray
    best_thr: float
    best_shd: int
    h_value_end: float
    history: List[Dict[str, Any]]


class CASTLE(object):
    """
    Modified CASTLE focused on Experiment 1:
    - augmented Lagrangian updates (rho, alpha) to enforce acyclicity
    - SHD-based selection over a threshold grid during training (if adj_true provided)
    - structured history logging returned from fit()
    """

    def __init__(
        self,
        num_train: int,
        lr: float = 1e-4,
        batch_size: int = 32,
        num_inputs: int = 1,
        num_outputs: int = 1,
        w_threshold: float = 0.0,
        n_hidden: int = 32,
        hidden_layers: int = 2,
        ckpt_file: str = "tmp.ckpt",
        standardize: bool = True,
        reg_lambda: float = 1.0,
        reg_beta: float = 5.0,
        DAG_min: float = 0.5,
        reconstruction_loss_weight: float = 10.0,
        dag_penalty_weight: float = 25.0,
        supervised_loss_weight: float = 0.0,
        max_steps: int = 200,
        saves: int = 10,
        patience: int = 30,
        seed: int = 1,
    ):
        self.w_threshold = w_threshold
        self.DAG_min = DAG_min
        self.learning_rate = lr
        self.reg_lambda = reg_lambda
        self.reg_beta = reg_beta
        self.batch_size = batch_size
        self.num_inputs = num_inputs
        self.n_hidden = n_hidden
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.dag_penalty_weight = dag_penalty_weight
        self.supervised_loss_weight = supervised_loss_weight

        self.max_steps = max_steps
        self.saves = saves
        self.patience = patience

        self.metric = mean_squared_error

        # placeholders
        self.X = tf.placeholder("float", [None, self.num_inputs])
        self.y = tf.placeholder("float", [None, 1])
        self.rho = tf.placeholder("float", [1, 1])
        self.alpha = tf.placeholder("float", [1, 1])
        self.keep_prob = tf.placeholder("float")
        self.Lambda = tf.placeholder("float")
        self.noise = tf.placeholder("float")
        self.is_train = tf.placeholder(tf.bool, name="is_train")

        # subset indicator
        self.sample = tf.placeholder(tf.int32, [self.num_inputs])

        # weights & biases
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.weights = {}
        self.biases = {}

        for i in range(self.num_inputs):
            self.weights[f"w_h0_{i}"] = tf.Variable(tf.random_normal([self.num_inputs, self.n_hidden], seed=seed) * 0.01)
            self.weights[f"out_{i}"] = tf.Variable(tf.random_normal([self.n_hidden, self.num_outputs], seed=seed))
            self.biases[f"b_h0_{i}"] = tf.Variable(tf.random_normal([self.n_hidden], seed=seed) * 0.01)
            self.biases[f"out_{i}"] = tf.Variable(tf.random_normal([self.num_outputs], seed=seed))

        # shared layer
        self.weights["w_h1"] = tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden], seed=seed))
        self.biases["b_h1"] = tf.Variable(tf.random_normal([self.n_hidden], seed=seed))

        self.hidden_h0 = {}
        self.hidden_h1 = {}
        self.out_layer = {}
        self.Out_0 = []

        self.mask = {}
        self.activation = tf.nn.relu

        for i in range(self.num_inputs):
            indices = [i] * self.n_hidden
            self.mask[str(i)] = tf.transpose(tf.one_hot(indices, depth=self.num_inputs, on_value=0.0, off_value=1.0, axis=-1))

            self.weights[f"w_h0_{i}"] = self.weights[f"w_h0_{i}"] * self.mask[str(i)]
            self.hidden_h0[f"nn_{i}"] = self.activation(tf.add(tf.matmul(self.X, self.weights[f"w_h0_{i}"]), self.biases[f"b_h0_{i}"]))
            self.hidden_h1[f"nn_{i}"] = self.activation(tf.add(tf.matmul(self.hidden_h0[f"nn_{i}"], self.weights["w_h1"]), self.biases["b_h1"]))
            self.out_layer[f"nn_{i}"] = tf.matmul(self.hidden_h1[f"nn_{i}"], self.weights[f"out_{i}"]) + self.biases[f"out_{i}"]
            self.Out_0.append(self.out_layer[f"nn_{i}"])

        self.Out = tf.concat(self.Out_0, axis=1)

        self.optimizer_subset = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # supervised loss (only used if supervised_loss_weight > 0)
        self.supervised_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.out_layer["nn_0"] - self.y), axis=1), axis=0)

        # build W (group norms of input-layer weights)
        self.W_0 = []
        for i in range(self.num_inputs):
            self.W_0.append(tf.sqrt(tf.reduce_sum(tf.square(self.weights[f"w_h0_{i}"]), axis=1, keepdims=True)))
        self.W = tf.concat(self.W_0, axis=1)

        # acyclicity constraint (NOTEARS-style trace exp approximation via truncated power series)
        d = tf.cast(self.X.shape[1], tf.float32)
        coff = 1.0
        Z = tf.multiply(self.W, self.W)

        dag_l = tf.cast(d, tf.float32)
        Z_in = tf.eye(d)
        for i in range(1, 10):
            Z_in = tf.matmul(Z_in, Z)
            dag_l += 1.0 / coff * tf.linalg.trace(Z_in)
            coff = coff * (i + 1)
        self.h = dag_l - tf.cast(d, tf.float32)

        # reconstruction residuals
        self.R = self.X - self.Out
        self.average_loss = 0.5 / num_train * tf.reduce_sum(tf.square(self.R))

        # group lasso
        L1_loss = 0.0
        for i in range(self.num_inputs):
            w_1 = tf.slice(self.weights[f"w_h0_{i}"], [0, 0], [i, -1])
            w_2 = tf.slice(self.weights[f"w_h0_{i}"], [i + 1, 0], [-1, -1])
            L1_loss += tf.reduce_sum(tf.norm(w_1, axis=1)) + tf.reduce_sum(tf.norm(w_2, axis=1))

        _, subset_R = tf.dynamic_partition(tf.transpose(self.R), partitions=self.sample, num_partitions=2)
        subset_R = tf.transpose(subset_R)

        self.mse_loss_subset = (
            tf.cast(self.num_inputs, tf.float32)
            / tf.cast(tf.reduce_sum(self.sample), tf.float32)
            * tf.reduce_sum(tf.square(subset_R))
        )

        dag_penalty = 0.5 * self.rho * self.h * self.h + self.alpha * self.h

        self.regularization_loss_subset = (
            self.reconstruction_loss_weight * self.mse_loss_subset
            + self.reg_beta * L1_loss
            + self.dag_penalty_weight * dag_penalty
        )

        if self.supervised_loss_weight > 0:
            self.regularization_loss_subset += self.supervised_loss_weight * self.supervised_loss

        self.loss_op_dag = self.optimizer_subset.minimize(self.regularization_loss_subset)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        try:
            from tensorflow.python.client import device_lib
            gpus = [d.name for d in device_lib.list_local_devices() if d.device_type == 'GPU']
            # best-effort: not all clusters expose device names the same way
            print('Detected TF GPUs:', gpus)
        except Exception:
            pass

        self.saver = tf.train.Saver(var_list=tf.global_variables())
        self.tmp = ckpt_file

    def __del__(self):
        try:
            tf.reset_default_graph()
            if hasattr(self, "sess"):
                self.sess.close()
        except Exception:
            pass

    def val_loss(self, X: np.ndarray) -> float:
        # validation uses all variables
        one_hot_sample = [1] * self.num_inputs
        y = np.expand_dims(X[:, 0], -1)
        return float(
            self.sess.run(
                self.regularization_loss_subset,
                feed_dict={
                    self.X: X,
                    self.y: y,
                    self.sample: one_hot_sample,
                    self.keep_prob: 1.0,
                    self.rho: np.array([[1.0]]),
                    self.alpha: np.array([[0.0]]),
                    self.Lambda: self.reg_lambda,
                    self.is_train: False,
                    self.noise: 0.0,
                },
            )
        )

    def get_adjacency(self, X_batch: np.ndarray) -> np.ndarray:
        y_batch = np.expand_dims(X_batch[:, 0], -1)
        feed = {
            self.X: X_batch,
            self.y: y_batch,
            self.rho: np.array([[1.0]]),
            self.alpha: np.array([[0.0]]),
            self.keep_prob: 1.0,
            self.Lambda: self.reg_lambda,
            self.noise: 0.0,
            self.is_train: False,
        }
        return self.sess.run(self.W, feed_dict=feed)

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        subset_nodes: int,
        adj_true: Optional[np.ndarray] = None,
        shd_threshold_grid: Optional[Sequence[float]] = None,
        rho_init: float = 1.0,
        alpha_init: float = 0.0,
        rho_multiplier: float = 10.0,
        dag_tol: float = 1e-8,
        logger: Optional[Any] = None,
        training_jsonl_path: Optional[str] = None,
        seed: int = 1,
    ) -> FitResult:
        """
        Returns FitResult with full history.
        If adj_true is provided, we select best W by SHD over threshold grid each epoch.
        """
        if shd_threshold_grid is None:
            shd_threshold_grid = [0.05, 0.1, 0.15, 0.2, 0.3]

        rng = np.random.default_rng(seed)

        rho_i = np.array([[float(rho_init)]])
        alpha_i = np.array([[float(alpha_init)]])

        best_val = float("inf")
        best_step = -1
        patience_count = 0

        # SHD selection
        best_shd = 10**18
        best_W = None
        best_thr = float(shd_threshold_grid[0])

        history: List[Dict[str, Any]] = []

        # jsonl logger
        jf = None
        if training_jsonl_path is not None:
            jf = open(training_jsonl_path, "w", encoding="utf-8")

        def log(msg: str) -> None:
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)

        for step in range(1, self.max_steps + 1):
            # train mini-batches
            n_batches = max(1, X_train.shape[0] // self.batch_size)
            for _ in range(n_batches):
                idxs = rng.choice(X_train.shape[0], size=self.batch_size, replace=False) if X_train.shape[0] >= self.batch_size else rng.choice(X_train.shape[0], size=self.batch_size, replace=True)
                batch_x = X_train[idxs]
                batch_y = np.expand_dims(batch_x[:, 0], -1)

                one_hot_sample = [0] * self.num_inputs
                subset_ = rng.choice(self.num_inputs, size=min(subset_nodes, self.num_inputs), replace=False)
                for j in subset_:
                    one_hot_sample[int(j)] = 1

                self.sess.run(
                    self.loss_op_dag,
                    feed_dict={
                        self.X: batch_x,
                        self.y: batch_y,
                        self.sample: one_hot_sample,
                        self.keep_prob: 1.0,
                        self.rho: rho_i,
                        self.alpha: alpha_i,
                        self.Lambda: self.reg_lambda,
                        self.is_train: True,
                        self.noise: 0.0,
                    },
                )

            # end-of-epoch eval
            h_value = float(
                self.sess.run(
                    self.h,
                    feed_dict={
                        self.X: X_train,
                        self.y: np.expand_dims(X_train[:, 0], -1),
                        self.keep_prob: 1.0,
                        self.rho: rho_i,
                        self.alpha: alpha_i,
                        self.is_train: False,
                        self.noise: 0.0,
                    },
                )
            )
            val_loss = self.val_loss(X_val)

            # augmented Lagrangian updates
            if h_value > self.DAG_min:
                rho_i *= rho_multiplier
            alpha_i += rho_i * h_value

            # SHD selection (if adj_true provided)
            if adj_true is not None:
                W_now = self.get_adjacency(X_train)
                # choose threshold that minimizes SHD on adjacency matrices directly
                for thr in shd_threshold_grid:
                    A_hat = (W_now > thr).astype(float)
                    # SHD proxy on adjacency matrix:
                    shd = int(np.sum(np.abs(A_hat - adj_true)))
                    if shd < best_shd:
                        best_shd = shd
                        best_W = W_now.copy()
                        best_thr = float(thr)

            row = {
                "step": step,
                "val_loss": float(val_loss),
                "h_value": float(h_value),
                "rho": float(rho_i.ravel()[0]),
                "alpha": float(alpha_i.ravel()[0]),
                "best_shd": int(best_shd) if adj_true is not None else None,
                "best_thr": float(best_thr) if adj_true is not None else None,
            }
            history.append(row)
            if jf is not None:
                jf.write(json.dumps(row) + "\n")
                jf.flush()

            log(f"Epoch {step:04d} | val_loss={val_loss:.6f} | h={h_value:.6e} | rho={rho_i.ravel()[0]:.3g} | alpha={alpha_i.ravel()[0]:.3g}")

            # early stopping based on val_loss
            if val_loss < best_val - 1e-12:
                best_val = val_loss
                best_step = step
                patience_count = 0
                # checkpoint
                if step >= self.saves:
                    self.saver.save(self.sess, self.tmp)
            else:
                patience_count += 1
                if patience_count > self.patience:
                    log("Early stopping triggered.")
                    break

        # restore best checkpoint (if any)
        try:
            self.saver.restore(self.sess, self.tmp)
        except Exception:
            pass

        if best_W is None:
            best_W = self.get_adjacency(X_train)

        if jf is not None:
            jf.close()

        return FitResult(
            best_W=best_W,
            best_thr=best_thr,
            best_shd=int(best_shd) if adj_true is not None else -1,
            h_value_end=float(history[-1]["h_value"]) if history else float("nan"),
            history=history,
        )
