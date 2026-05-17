import copy
import json
import os
import random
import warnings

import numpy as np
import optuna
import torch
from optuna.exceptions import TrialPruned
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from load_set import ber_to_class


class MLPClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout, activation_name="relu", num_classes=3):
        super().__init__()

        activation_map = {
            "relu": torch.nn.ReLU,
            "leaky_relu": lambda: torch.nn.LeakyReLU(negative_slope=0.01),
            "elu": torch.nn.ELU,
            "gelu": torch.nn.GELU,
        }
        activation_factory = activation_map.get(activation_name, torch.nn.ReLU)

        layers = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(previous_size, hidden_size))
            layers.append(activation_factory())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            previous_size = hidden_size

        layers.append(torch.nn.Linear(previous_size, num_classes))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        
def available_model_families():
    return ["xgb", "lightgbm", "catboost", "extratrees", "mlp"]


def build_tree_model(trial, family, seed):
    if family == "xgb":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=trial.suggest_int("xgb_n_estimators", 100, 1200),
            max_depth=trial.suggest_int("xgb_max_depth", 2, 8),
            learning_rate=trial.suggest_float("xgb_learning_rate", 1e-3, 0.3, log=True),
            gamma=trial.suggest_float("xgb_gamma", 0.0, 5.0),
            subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("xgb_reg_lambda", 1e-3, 20.0, log=True),
            min_child_weight=trial.suggest_float("xgb_min_child_weight", 1e-2, 20.0, log=True),
            random_state=seed,
            n_jobs=-1,
            tree_method=trial.suggest_categorical("xgb_tree_method", ["hist", "auto"]),
            eval_metric="mlogloss",
        )

    if family == "lightgbm":
        return LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=trial.suggest_int("lgbm_n_estimators", 100, 1500),
            num_leaves=trial.suggest_int("lgbm_num_leaves", 16, 256),
            max_depth=trial.suggest_int("lgbm_max_depth", -1, 12),
            learning_rate=trial.suggest_float("lgbm_learning_rate", 1e-3, 0.3, log=True),
            subsample=trial.suggest_float("lgbm_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("lgbm_colsample_bytree", 0.6, 1.0),
            min_child_samples=trial.suggest_int("lgbm_min_child_samples", 5, 100),
            reg_lambda=trial.suggest_float("lgbm_reg_lambda", 1e-3, 20.0, log=True),
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )

    if family == "catboost":
        return CatBoostClassifier(
            loss_function="MultiClass",
            iterations=trial.suggest_int("cat_iterations", 100, 1500),
            depth=trial.suggest_int("cat_depth", 3, 10),
            learning_rate=trial.suggest_float("cat_learning_rate", 1e-3, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("cat_l2_leaf_reg", 1e-3, 50.0, log=True),
            random_strength=trial.suggest_float("cat_random_strength", 0.0, 5.0),
            bagging_temperature=trial.suggest_float("cat_bagging_temperature", 0.0, 1.0),
            border_count=trial.suggest_int("cat_border_count", 32, 255),
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )

    if family == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("et_n_estimators", 100, 1000),
            max_depth=trial.suggest_int("et_max_depth", 2, 30),
            min_samples_split=trial.suggest_int("et_min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("et_min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("et_max_features", ["sqrt", "log2", None]),
            bootstrap=trial.suggest_categorical("et_bootstrap", [False, True]),
            class_weight=trial.suggest_categorical("et_class_weight", [None, "balanced"]),
            random_state=seed,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown tree model family: {family}")


def train_mlp(
    trial,
    x_train,
    y_train,
    x_val,
    y_val,
    seed,
    input_size,
    num_classes,
    batch_size,
    epochs,
    patience,
    device,
):
    num_layers = trial.suggest_int("mlp_num_layers", 1, 4)
    hidden_sizes = []
    for i in range(num_layers):
        hidden_sizes.append(trial.suggest_categorical(f"mlp_units_l{i}", [32, 64, 128, 256]))

    activation = trial.suggest_categorical("mlp_activation", ["relu", "leaky_relu", "elu", "gelu"])
    dropout = trial.suggest_float("mlp_dropout", 0.0, 0.5)
    lr = trial.suggest_float("mlp_lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("mlp_weight_decay", 1e-6, 1e-2, log=True)
    batch_norm = trial.suggest_categorical("mlp_batch_norm", [False, True])

    layers = []
    previous_size = input_size
    activation_map = {
        "relu": torch.nn.ReLU,
        "leaky_relu": lambda: torch.nn.LeakyReLU(negative_slope=0.01),
        "elu": torch.nn.ELU,
        "gelu": torch.nn.GELU,
    }
    activation_factory = activation_map.get(activation, torch.nn.ReLU)

    for hidden_size in hidden_sizes:
        layers.append(torch.nn.Linear(previous_size, hidden_size))
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(hidden_size))
        layers.append(activation_factory())
        if dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        previous_size = hidden_size

    layers.append(torch.nn.Linear(previous_size, num_classes))
    model = torch.nn.Sequential(*layers).to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_tensor_x = torch.from_numpy(x_train)
    train_tensor_y = torch.from_numpy(y_train)
    val_tensor_x = torch.from_numpy(x_val)
    val_tensor_y = torch.from_numpy(y_val)

    train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        drop_last=batch_norm,
    )

    best_state = None
    best_score = float("-inf")
    epochs_since_improvement = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(val_tensor_x.to(device).float())
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_score = f1_score(y_val, preds, average="macro", zero_division=0)

        if val_score > best_score + 1e-8:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_score


def fit_tree_model(model, family, x_train, y_train, x_val, y_val, sample_weight):
    if family in {"xgb", "lightgbm", "catboost"}:
        fit_kwargs = {"eval_set": [(x_val, y_val)]}
        if family == "catboost":
            fit_kwargs["verbose"] = False
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(x_train, y_train, **fit_kwargs)
    else:
        if sample_weight is not None:
            model.fit(x_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(x_train, y_train)

    preds = model.predict(x_val)
    return f1_score(y_val, preds, average="macro", zero_division=0)


def run_optuna_classifier(
    x_array,
    y_array,
    feature_columns,
    lower_thres=-5.5,
    upper_thres=-2.5,
    logBER=True,
    n_trials=30,
    seed=42,
    study_name="deepber_classifier_optuna",
    storage=None,
    cv_folds=3,
    mlp_epochs=60,
    mlp_patience=10,
):
    set_seed(seed)

    if lower_thres >= upper_thres:
        raise ValueError("lower_thres must be smaller than upper_thres.")

    print(f"[optuna-classifier] Starting study '{study_name}' with n_trials={n_trials}, seed={seed}")
    print(f"[optuna-classifier] Loaded dataset: x_shape={x_array.shape}, y_shape={y_array.shape}, features={len(feature_columns)}")

    y_labels = ber_to_class(y_array, lower_thres=lower_thres, upper_thres=upper_thres, logBER=logBER)
    class_counts = np.bincount(y_labels, minlength=3)
    print(
        "[optuna-classifier] Class distribution: "
        f"feasible={class_counts[0]}, uncertain={class_counts[1]}, unfeasible={class_counts[2]}"
    )

    min_class_count = int(class_counts[class_counts > 0].min()) if np.any(class_counts > 0) else 0
    if min_class_count < 2:
        raise ValueError("At least two samples are required for every present class.")

    effective_cv_folds = min(cv_folds, min_class_count)
    if effective_cv_folds < 2:
        raise ValueError("cv_folds is too large for the available per-class sample counts.")
    if effective_cv_folds != cv_folds:
        warnings.warn(
            f"Reducing cv_folds from {cv_folds} to {effective_cv_folds} to preserve stratification.",
            RuntimeWarning,
        )

    input_size = len(feature_columns)
    skf = StratifiedKFold(n_splits=effective_cv_folds, shuffle=True, random_state=seed)
    available_families = available_model_families()
    print(f"[optuna-classifier] Available model families: {available_families}")
    print(f"[optuna-classifier] Using {effective_cv_folds}-fold stratified cross validation")

    def objective(trial: optuna.trial.Trial):
        family = trial.suggest_categorical("model_family", available_families)
        print(f"[optuna-classifier] Trial {trial.number}: model_family={family}")

        fold_scores = []

        for fold_index, (train_idx, val_idx) in enumerate(skf.split(x_array, y_labels), start=1):
            x_train = x_array[train_idx]
            y_train = y_labels[train_idx]
            x_val = x_array[val_idx]
            y_val = y_labels[val_idx]

            if family == "mlp":
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train).astype(np.float32)
                x_val = scaler.transform(x_val).astype(np.float32)
                batch_size = trial.suggest_categorical("mlp_batch_size", [16, 32, 64])
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model, fold_score = train_mlp(
                    trial,
                    x_train.astype(np.float32),
                    y_train.astype(np.int64),
                    x_val.astype(np.float32),
                    y_val.astype(np.int64),
                    seed,
                    input_size,
                    num_classes=3,
                    batch_size=batch_size,
                    epochs=mlp_epochs,
                    patience=mlp_patience,
                    device=device,
                )
                _ = model
            else:
                sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
                model = build_tree_model(trial, family, seed)
                fold_score = fit_tree_model(model, family, x_train, y_train, x_val, y_val, sample_weight)

            fold_scores.append(fold_score)
            cv_mean_score = float(np.mean(fold_scores))
            print(
                f"[optuna-classifier] Trial {trial.number}: fold {fold_index}/{effective_cv_folds}, "
                f"fold_score={fold_score:.6f}, cv_mean_score={cv_mean_score:.6f}"
            )

            trial.report(cv_mean_score, fold_index - 1)
            if trial.should_prune():
                print(f"[optuna-classifier] Trial {trial.number}: pruned after fold {fold_index}")
                raise TrialPruned()

        final_score = float(np.mean(fold_scores))
        print(f"[optuna-classifier] Trial {trial.number}: completed with cv_score={final_score:.6f}")
        return final_score

    if storage:
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(),
            storage=storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(),
        )

    print("[optuna-classifier] Beginning optimization...")
    study.optimize(objective, n_trials=n_trials)

    out = {
        "best_value": study.best_value,
        "best_params": study.best_params,
    }

    out_path = os.path.join(os.getcwd(), "optuna_classifier_study_result.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    return study
