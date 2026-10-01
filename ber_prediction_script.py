from prediction.optuna_tuner import run_optuna

import torch
from torch import nn
from load_set import create_dataloader, load_csv_dataset, create_arrays
from classification.classifier import DeepBERClassifier
from classification.test_classifier_config import test_classifier_configuration
from prediction.predictor import DeepBERPredictor
from prediction.test_predictor_config import test_predictor_configuration, ber_vs_length_test
from classification.ber_to_class import ber_to_class
import numpy as np
from classification.optuna_tuner import run_optuna_classifier
from sklearn.utils.class_weight import compute_class_weight

torch.manual_seed(42)

# ============================================= Initializing Dataset ============================================= #

ber_og_csv_names = ["ber_og_database.csv"]
wrst_case_csv_names = ["wrst_case_ber_database1.csv", "wrst_case_ber_database2.csv", "wrst_case_ber_database3.csv"]
prbs_case_csv_names = ["prbs_case_database1.csv", "prbs_case_database2.csv"]
combo_csv_names = ["prbs_case_database1.csv", "prbs_case_database2.csv", "wrst_case_ber_database1.csv", "wrst_case_ber_database2.csv",
                   "wrst_case_ber_database3.csv"]
extnd_csv_names = ["prbs_case_database1.csv", "prbs_case_database2.csv", "wrst_case_ber_database1.csv", "wrst_case_ber_database2.csv",
                "wrst_case_ber_database3.csv", "delay_csv_database_extnd.csv"]

x_array, y_array, _, _, thresholds, feature_columns = create_arrays(
                                                                    csv_names=extnd_csv_names, # Change to desired dataset
                                                                    target_columns=["BER"],
                                                                    thresholds=(10**-5.5, 10**-2.5),
                                                                    manipulate_features=True,
                                                                )


# =================================================== Training and Testing ================================================== #
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 16
prediction_dataloader = create_dataloader(
            x_array,
            y_array,
            logBER=True,
            batch_size=batch_size,
            seed=42,
            ber_interval=thresholds,
            standard_scale=True
        )

predictor = DeepBERPredictor(
        input_size=len(feature_columns),  
        hidden=[128, 32, 48],
        activation_fn=nn.GELU(),
        logBER=True,
        batch_norm=False,
        dropout=0.244,
    ).to(device)

learning_rate = 0.0058
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=39)

test_predictor_configuration(
    title="DeepBER BER prediction",
    device=device,
    model=predictor,
    dataloader=prediction_dataloader,
    learning_rate=learning_rate,
    batch_size=batch_size,
    criterion=criterion,
    optimizer=optimizer,
    # scheduler=None,
    epochs=240,
    early_stopping=True,
    patience=10,
    training_curves=True,
    predicted_vs_actual=True,
    # error_distribution=True,
    # error_vs_feature=feature_columns,
    # feature_columns=feature_columns
)

