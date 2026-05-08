import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

def loader_to_numpy(data_loader):
    # Convert a torch DataLoader into NumPy arrays for non-torch models.
    tensors = data_loader.dataset.tensors
    features = tensors[0].cpu().numpy()
    labels = tensors[1].cpu().numpy().reshape(-1)
    return features, labels


def train_xgb_loop(
    model,
    train_data,
    val_data=None,
    label_transform=None,
    lower_thres=10**(-5.5),
    upper_thres=10**(-2.5),
    weight=False,
):
    # Train an XGBoost classifier on train_data and optional validation data.
    #
    # Args:
    # - model: Configured XGBoost classifier instance
    # - train_data: Training DataLoader
    # - val_data: Optional validation DataLoader
    # - label_transform: Optional callable to convert labels
    # Returns:
    # - model: Trained model

    x_train, y_train = loader_to_numpy(train_data)

    if label_transform is not None:
        y_train = label_transform(y_train, lower_thres=lower_thres, upper_thres=upper_thres)

    fit_kwargs = {}

    if weight is True:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        fit_kwargs["sample_weight"] = sample_weight

    if val_data is not None:
        x_val, y_val = loader_to_numpy(val_data)
        if label_transform is not None:
            y_val = label_transform(y_val, lower_thres=lower_thres, upper_thres=upper_thres)
        fit_kwargs["eval_set"] = [(x_val, y_val)]
        fit_kwargs["verbose"] = False

    model.fit(x_train, y_train, **fit_kwargs)
    return model


def test_xgb_loop(model, data, label_transform=None, lower_thres=10**(-5.5), upper_thres=10**(-2.5)):
    # Evaluate an XGBoost classifier.
    #
    # Args:
    # - model: Trained XGBoost classifier
    # - data: Validation or test DataLoader
    # - label_transform: Optional callable to convert labels
    # Returns:
    # - accuracy: Overall accuracy
    # - class_accuracy: Dict with per-class accuracy for classes 0,1,2
    # - preds: Predicted class labels
    # - labels: True class labels

    features, labels = loader_to_numpy(data)
    if label_transform is not None:
        labels = label_transform(labels, lower_thres=lower_thres, upper_thres=upper_thres)

    preds = model.predict(features)
    accuracy = float(np.mean(preds == labels))

    class_accuracy = {}
    for class_id in [0, 1, 2]:
        class_mask = labels == class_id
        if np.any(class_mask):
            class_accuracy[class_id] = float(np.mean(preds[class_mask] == labels[class_mask]))
        else:
            class_accuracy[class_id] = float("nan")

    return accuracy, class_accuracy, preds, labels