import numpy as np
from classification.classifier_loops import train_xgb_loop, test_xgb_loop

def ber_to_class(ber_values, lower_thres=10**(-5.5), upper_thres=10**(-2.5) ):
    # Class mapping:
    # 0 -> BER < lower_thres
    # 1 -> lower_thres <= BER <= upper_thres
    # 2 -> BER > upper_thres
    labels = np.ones_like(ber_values, dtype=np.int64)
    labels[ber_values < lower_thres] = 0
    labels[ber_values > upper_thres] = 2
    return labels

def test_classifier_configuration(title, model, dataloader,):
    train_data, val_data, test_data = dataloader

    trained_model = train_xgb_loop(model, train_data, val_data=val_data, label_transform=ber_to_class)

    test_acc, _, _, _ = test_xgb_loop(trained_model, test_data, label_transform=ber_to_class)

    print(f"\nAccuracy: {test_acc:.4f}")

    return trained_model, test_acc

