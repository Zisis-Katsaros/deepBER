import numpy as np
from classification.classifier_loops import train_xgb_loop, test_xgb_loop
from visualization import plot_confusion_matrix

def ber_to_class(ber_values, lower_thres, upper_thres):
    # Class mapping:
    # 0 -> BER < lower_thres
    # 1 -> lower_thres <= BER <= upper_thres
    # 2 -> BER > upper_thres
    labels = np.ones_like(ber_values, dtype=np.int64)
    labels[ber_values < lower_thres] = 0
    labels[ber_values > upper_thres] = 2
    return labels

def test_classifier_configuration(
    title,
    model,
    dataloader,
    lower_thres=10**(-5.5),
    upper_thres=10**(-2.5),
    weight=False,
    confusion_matrix=False,
    class_names=None, 
):
    train_data, val_data, test_data = dataloader

    trained_model = train_xgb_loop(
        model,
        train_data,
        val_data=val_data,
        label_transform=ber_to_class,
        lower_thres=lower_thres,
        upper_thres=upper_thres,
        weight=weight,
    )

    test_acc, _, test_preds, test_labels = test_xgb_loop(trained_model, test_data, label_transform=ber_to_class, lower_thres=lower_thres, 
                                                         upper_thres=upper_thres)

    print(f"\nAccuracy: {test_acc:.4f}")

    if confusion_matrix:
        if class_names is None:
            class_names = ["Feasible", "Uncertain", "Infeasible"]
        plot_confusion_matrix(test_labels, test_preds, title=title, class_names=class_names)

    return trained_model, test_acc

