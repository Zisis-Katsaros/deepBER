from xgboost import XGBClassifier

def xgb_classifier(n_estimators=350, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, seed=42, eval_metric="mlogloss"):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=seed,
        eval_metric=eval_metric,
    )
