from xgboost import XGBClassifier

def xgb_classifier(n_estimators=1000, max_depth=3, learning_rate=0.01, gamma=1.5, subsample=0.8, colsample_bytree=0.9, reg_lambda=1.0, seed=42, eval_metric="mlogloss"):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=seed,
        eval_metric=eval_metric,
    )
