import torch
from torch import nn
from load_set import create_dataloader, create_arrays
from classification.classifier import DeepBERClassifier
from classification.test_classifier_config import test_classifier_configuration
import numpy as np
from classification.optuna_tuner import run_optuna_classifier
from sklearn.utils.class_weight import compute_class_weight

torch.manual_seed(42)

# ============================================= Initializing Dataset ============================================= #
bin_classification = True
logBER = False

ber_og_csv_names = ["ber_og_database.csv"]
wrst_case_csv_names = ["wrst_case_ber_database1.csv", "wrst_case_ber_database2.csv", "wrst_case_ber_database3.csv"]
prbs_case_csv_names = ["prbs_case_database1.csv", "prbs_case_database2.csv"]
combo_csv_names = ["prbs_case_database1.csv", "prbs_case_database2.csv", "wrst_case_ber_database1.csv", "wrst_case_ber_database2.csv",
                   "wrst_case_ber_database3.csv"]
extnd_csv_names = ["prbs_case_database1.csv", "prbs_case_database2.csv", "wrst_case_ber_database1.csv", "wrst_case_ber_database2.csv",
                "wrst_case_ber_database3.csv", "delay_csv_database_extnd.csv"]

x_array, y_array, y_array_log, y_classes, thresholds, feature_columns = create_arrays(
                                                                            csv_names=extnd_csv_names, # Change to desired dataset
                                                                            target_columns=["BER"],
                                                                            thresholds=(10**-5.5, 10**-2.5),
                                                                            manipulate_features=True,
                                                                            binary_classification=bin_classification,
                                                                        )


# =================================================== Training and Testing ================================================== #
device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = 2 if bin_classification else 3

lower_thres, upper_thres = -5.5, -2.5

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=y_classes.ravel(), # Flatten the array to 1D
)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

lr = 0.00379
weight_decay = 5.045e-5

num_of_runs = 5
total_acc = 0.0
total_f1 = 0.0
for k in range (num_of_runs):
    seed = 42 + k

    # Create a fresh model per run so each seed is evaluated independently.
    classifier = DeepBERClassifier(
        input_size=len(feature_columns),
        num_classes=num_classes,
        hidden=[64, 32, 64, 48],
        activation_fn=nn.GELU(),
        logBER=logBER,
        batch_norm=False,
        dropout=0.241,
    )
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)

    classifier_dataloader = create_dataloader(
    x_array,
    y_classes,
    logBER=logBER,
    batch_size=16,
    seed=seed,
    standard_scale=True,
    split_method="lhs"
)
    
    print(f"\n\n--- Run {k+1} with seed {seed} ---\n\n")

    _, test_acc, test_f1 = test_classifier_configuration(
        title="MLP Classifier",
        model=classifier,
        dataloader=classifier_dataloader,
        lower_thres=lower_thres,
        upper_thres=upper_thres,
        device=device,
        learning_rate=lr,
        criterion=criterion,
        optimizer=optimizer,
        epochs=60,
        early_stopping=True,
        patience=10,
        confusion_matrix=False,
    )

    total_acc += test_acc
    total_f1 += test_f1

avg_acc = total_acc / num_of_runs
avg_f1 = total_f1 / num_of_runs

print(f"\n\nAverage Test Accuracy over {num_of_runs} runs: {avg_acc*100:.2f}%")
print(f"Average Test F1 Score over {num_of_runs} runs: {avg_f1:.4f}")
