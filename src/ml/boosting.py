from collections import defaultdict
from typing import Optional

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class Boosting:
    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        early_stopping_rounds: Optional[int] = 0,
        subsample: float = 1.0,
        bagging_temperature: float = 1.0,
        bootstrap_type: Optional[str] = None,
        a: Optional[float] = None,
        b: Optional[float] = None,
        use_goss: bool = False,
        use_bootstrap: bool = False,
        use_dart: bool = False,
        dart_rate: float = 0.05,
        random_state: Optional[int] = None,
    ):
        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.gammas = []
        self.history = defaultdict(list)
        self.feature_names_ = None

        self.early_stopping_rounds = early_stopping_rounds
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.a = a
        self.b = b
        self.use_goss = use_goss
        self.use_bootstrap = use_bootstrap
        self.use_dart = use_dart
        self.dart_rate = dart_rate
        self.rng = np.random.RandomState(random_state)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

    def loss_fn(self, y, z):
        return np.log(1 + np.exp(-np.clip(y * z, -30, 30))).mean()

    def loss_derivative(self, y, z):
        return y * (1 - self.sigmoid(-y * z))

    def _get_bootstrap_indices(self, X):
        n_samples = X.shape[0]
        if self.bootstrap_type == "Bernoulli":
            mask = self.rng.rand(n_samples) < self.subsample
            idx = np.where(mask)[0]
            return idx if len(idx) > 0 else np.arange(n_samples)
        if self.bootstrap_type == "Bayesian":
            weights = -np.log(self.rng.rand(n_samples)) * self.bagging_temperature
            probs = weights / weights.sum()
            return self.rng.choice(n_samples, size=max(1, int(self.subsample * n_samples)), replace=True, p=probs)
        return np.arange(n_samples)

    def _goss_sampling(self, gradients):
        n = len(gradients)
        n_top = max(1, int(self.a * n))
        abs_grads = np.abs(gradients)
        sorted_idx = np.argsort(-abs_grads)
        top_idx = sorted_idx[:n_top]
        rest_idx = sorted_idx[n_top:]
        n_rest = max(1, int(self.b * len(rest_idx))) if len(rest_idx) > 0 else 0
        if n_rest > 0:
            sampled_rest = self.rng.choice(rest_idx, min(n_rest, len(rest_idx)), replace=False)
        else:
            sampled_rest = np.array([], dtype=int)
        selected = np.concatenate([top_idx, sampled_rest]).astype(int)
        weights = np.ones_like(gradients)
        if len(sampled_rest) > 0:
            weights[sampled_rest] = (1 - self.a) / self.b
        return selected, gradients[selected] * weights[selected]

    def _find_optimal_gamma(self, y, old_predictions, new_predictions):
        gammas = np.linspace(0, 1, 50)
        losses = [self.loss_fn(y, old_predictions + g * new_predictions) for g in gammas]
        return gammas[int(np.argmin(losses))]

    def _standard_step(self, X, y_transformed, old_predictions):
        gradients = self.loss_derivative(y_transformed, old_predictions)

        if self.use_goss:
            indices, residuals = self._goss_sampling(gradients)
        elif self.use_bootstrap:
            indices = self._get_bootstrap_indices(X)
            residuals = gradients[indices]
        else:
            indices = np.arange(len(gradients))
            residuals = gradients

        model = self.base_model_class(**self.base_model_params)
        model.fit(X[indices], residuals)
        new_tree_preds = model.predict(X)

        gamma = self._find_optimal_gamma(y_transformed, old_predictions, new_tree_preds)
        self.models.append(model)
        self.gammas.append(gamma)
        return old_predictions + self.learning_rate * gamma * new_tree_preds

    def _dart_step(self, X, y_transformed, old_predictions):
        n_trees = len(self.models)
        if n_trees == 0:
            return self._standard_step(X, y_transformed, old_predictions)

        n_drop = max(1, int(n_trees * self.dart_rate))
        drop_indices = self.rng.choice(n_trees, size=n_drop, replace=False)

        pred_without_dropped = np.zeros(X.shape[0])
        for i, (model, gamma) in enumerate(zip(self.models, self.gammas)):
            if i not in drop_indices:
                pred_without_dropped += self.learning_rate * gamma * model.predict(X)

        gradients = self.loss_derivative(y_transformed, pred_without_dropped)
        model = self.base_model_class(**self.base_model_params)
        model.fit(X, gradients)
        new_tree_preds = model.predict(X)

        gamma = self._find_optimal_gamma(y_transformed, pred_without_dropped, new_tree_preds)
        scale = 1.0 / (n_drop + 1.0)
        gamma *= scale

        for idx in drop_indices:
            self.gammas[idx] *= (n_drop / (n_drop + 1.0))

        self.models.append(model)
        self.gammas.append(gamma)
        return self._predict_raw(X)

    def fit(self, X, y, X_val=None, y_val=None, feature_names=None):
        self.feature_names_ = feature_names
        train_predictions = np.zeros(X.shape[0])
        y_transformed = np.where(y == 1, 1, -1)
        y_val_transformed = np.where(y_val == 1, 1, -1) if y_val is not None else None

        best_val_loss = float("inf")
        rounds_without_improve = 0

        for i in range(self.n_estimators):
            if self.use_dart:
                train_predictions = self._dart_step(X, y_transformed, train_predictions)
            else:
                train_predictions = self._standard_step(X, y_transformed, train_predictions)

            self.history["train_loss"].append(self.loss_fn(y_transformed, train_predictions))

            if X_val is not None:
                val_preds = self._predict_raw(X_val)
                val_loss = self.loss_fn(y_val_transformed, val_preds)
                self.history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    rounds_without_improve = 0
                else:
                    rounds_without_improve += 1

                if self.early_stopping_rounds and rounds_without_improve >= self.early_stopping_rounds:
                    break

        return self

    def _predict_raw(self, X):
        preds = np.zeros(X.shape[0])
        for model, gamma in zip(self.models, self.gammas):
            preds += self.learning_rate * gamma * model.predict(X)
        return preds

    def predict_proba(self, X):
        raw = self._predict_raw(X)
        proba = self.sigmoid(raw)
        return np.vstack([1 - proba, proba]).T

    def feature_importance(self):
        if not self.models:
            return {}
        n_features = self.models[0].feature_importances_.shape[0]
        totals = np.zeros(n_features)
        for model, gamma in zip(self.models, self.gammas):
            weight = abs(self.learning_rate * gamma)
            totals += weight * model.feature_importances_
        if totals.sum() > 0:
            totals /= totals.sum()
        names = self.feature_names_ or [f"f{i}" for i in range(n_features)]
        return dict(sorted(zip(names, totals), key=lambda kv: -kv[1]))