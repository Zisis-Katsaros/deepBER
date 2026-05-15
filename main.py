from prediction.optuna_tuner import run_optuna

import torch
from torch import nn
from load_set import create_dataloader, load_csv_dataset
from dataset_manipulation import extend_features, exclude_columns
from classification.classifier import xgb_classifier
from classification.test_classifier_config import test_classifier_configuration
from prediction.predictor import DeepBERPredictor
from prediction.test_predictor_config import test_predictor_configuration, ber_vs_length_test
import numpy as np

def main():
    torch.manual_seed(42)

    # ============================================= Initializing Dataset ============================================= #

    # 1. BER OG ------------------------------------------------------------------------------------
    ber_og = "ber_og_database.csv"
    x_array_ber_og, y_array_ber_og, feature_columns_ber_og = load_csv_dataset([ber_og], target_column="BER")

    eps = 1e-15 # To avoid log(0)
    y_array_ber_og_log = np.log10(np.clip(y_array_ber_og, eps, None)).astype(np.float32)

    x_array_ber_og, feature_columns_ber_og = extend_features(x_array_ber_og, feature_columns_ber_og, "width", "space", "/", "width_space_ratio")
    x_array_ber_og, feature_columns_ber_og = extend_features(x_array_ber_og, feature_columns_ber_og, "width", "metal_thickness", "*", "cross_sectional_area")
    x_array_ber_og, feature_columns_ber_og = extend_features(x_array_ber_og, feature_columns_ber_og, "gnd_width", "width", "/", "gnd_width_width_ratio")
    x_array_ber_og, feature_columns_ber_og = exclude_columns(x_array_ber_og, feature_columns_ber_og, columns_to_exclude=["delay"])

    batch_size_ber_og = 32
    gray_area_interval_ber_og = [10**-5.5, 10**-2.5]

    pred_dataloader_ber_og = create_dataloader(
        x_array_ber_og,
        y_array_ber_og,
        logBER=True,
        batch_size=batch_size_ber_og,
        seed=42,
        ber_interval=gray_area_interval_ber_og,
        standard_scale=True
    )

    # 2. WRST CASE SNR --------------------------------------------------------------------------------
    wrst_case_snr_names = ["wrst_case_snr_database1.csv", "wrst_case_snr_database3.csv"]
    x_array_wrst_snr, y_array_wrst_snr, feature_columns_wrst_snr = load_csv_dataset(wrst_case_snr_names, target_column="snr")

    x_array_wrst_snr, feature_columns_wrst_snr = extend_features(x_array_wrst_snr, feature_columns_wrst_snr, "width", "space", "/", "width_space_ratio")    
    x_array_wrst_snr, feature_columns_wrst_snr = extend_features(x_array_wrst_snr, feature_columns_wrst_snr, "width", "metal_thickness", "*", "cross_sectional_area")
    x_array_wrst_snr, feature_columns_wrst_snr = extend_features(x_array_wrst_snr, feature_columns_wrst_snr, "gnd_width", "width", "/", "gnd_width_width_ratio")
    x_array_wrst_snr, feature_columns_wrst_snr = exclude_columns(x_array_wrst_snr, feature_columns_wrst_snr, columns_to_exclude=["delay"])

    batch_size_wrst_snr = 8
    gray_area_interval_wrst_snr = [4.36, 6.55]

    pred_dataloader_wrst_snr = create_dataloader(
        x_array_wrst_snr,
        y_array_wrst_snr,
        logBER=False,
        batch_size=batch_size_wrst_snr,
        seed=42,
        ber_interval=gray_area_interval_wrst_snr,
        standard_scale=True
    )

    # 3. WRST CASE BER --------------------------------------------------------------------------------
    wrst_case_ber_names = ["wrst_case_ber_database1.csv", "wrst_case_ber_database2.csv", "wrst_case_ber_database3.csv"]

    x_array_wrst_ber, y_array_wrst_ber, feature_columns_wrst_ber = load_csv_dataset(wrst_case_ber_names, target_column="BER")
    
    eps = 1e-15 # To avoid log(0)
    y_array_wrst_ber_log = np.log10(np.clip(y_array_wrst_ber, eps, None)).astype(np.float32)

    x_array_wrst_ber, feature_columns_wrst_ber = extend_features(x_array_wrst_ber, feature_columns_wrst_ber, "width", "space", "/", "width_space_ratio")
    x_array_wrst_ber, feature_columns_wrst_ber = extend_features(x_array_wrst_ber, feature_columns_wrst_ber, "width", "metal_thickness", "*", "cross_sectional_area")
    x_array_wrst_ber, feature_columns_wrst_ber = extend_features(x_array_wrst_ber, feature_columns_wrst_ber, "gnd_width", "width", "/", "gnd_width_width_ratio")
    x_array_wrst_ber, feature_columns_wrst_ber = exclude_columns(x_array_wrst_ber, feature_columns_wrst_ber, columns_to_exclude=["delay"])

    batch_size_wrst_ber = 16
    gray_area_interval_wrst_ber = [10**-5.5, 10**-2.5]

    pred_dataloader_wrst_ber = create_dataloader(
        x_array_wrst_ber,
        y_array_wrst_ber,
        logBER=True,
        batch_size=batch_size_wrst_ber,
        seed=42,
        ber_interval=gray_area_interval_wrst_ber,
        standard_scale=True
    )

    # 4. PRBS CASE BER --------------------------------------------------------------------------------
    prbs_case_ber_names = ["prbs_case_database1.csv", "prbs_case_database2.csv"]

    x_array_prbs_ber, y_array_prbs_ber, feature_columns_prbs_ber = load_csv_dataset(prbs_case_ber_names, target_column="BER")

    eps = 1e-15 # To avoid log(0)
    y_array_prbs_ber_log = np.log10(np.clip(y_array_prbs_ber, eps, None)).astype(np.float32)

    x_array_prbs_ber, feature_columns_prbs_ber = extend_features(x_array_prbs_ber, feature_columns_prbs_ber, "width", "space", "/", "width_space_ratio")
    x_array_prbs_ber, feature_columns_prbs_ber = extend_features(x_array_prbs_ber, feature_columns_prbs_ber, "width", "metal_thickness", "*", "cross_sectional_area")
    x_array_prbs_ber, feature_columns_prbs_ber = extend_features(x_array_prbs_ber, feature_columns_prbs_ber, "gnd_width", "width", "/", "gnd_width_width_ratio")
    x_array_prbs_ber, feature_columns_prbs_ber = exclude_columns(x_array_prbs_ber, feature_columns_prbs_ber, columns_to_exclude=["delay"])

    batch_size_prbs_ber = 32
    gray_area_interval_prbs_ber = [10**-5.5, 10**-2.5]

    pred_dataloader_prbs_ber = create_dataloader(
        x_array_prbs_ber,
        y_array_prbs_ber,
        logBER=True,
        batch_size=batch_size_prbs_ber,
        seed=42,
        ber_interval=gray_area_interval_prbs_ber,
        standard_scale=True
    )

    # 5. WRST + PRBS COMBINED DATASET ---------------------------------------------------------------
    x_array_combined = np.concatenate([x_array_wrst_ber, x_array_prbs_ber], axis=0)
    y_array_combined = np.concatenate([y_array_wrst_ber, y_array_prbs_ber], axis=0)
    y_array_combined_log = np.concatenate([y_array_wrst_ber_log, y_array_prbs_ber_log], axis=0)
    feature_columns_combined = feature_columns_wrst_ber

    batch_size_combined = 16
    gray_area_interval_combined = [10**-5.5, 10**-2.5]

    pred_dataloader_combined = create_dataloader(
        x_array_combined,
        y_array_combined,
        logBER=True,
        batch_size=batch_size_combined,
        seed=42,
        ber_interval=gray_area_interval_combined,
        standard_scale=True
    )

    # =================================================== Classifier ================================================== #
    classifier = xgb_classifier(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.01,
        gamma=1.0,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        seed=42,
        eval_metric="mlogloss"
    )
    # lower_thres = gray_area_interval[0]
    # upper_thres = gray_area_interval[1]
    """
    test_classifier_configuration(
        title="XGBoost Baseline",
        model=classifier,
        dataloader=classifier_dataloader,
        lower_thres=lower_thres,
        upper_thres=upper_thres,
        confusion_matrix=True
    )
    """

    # =================================================== Predictor =================================================== #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = DeepBERPredictor(
        input_size=len(feature_columns_combined),
        hidden=[32, 16],
        activation_fn=nn.GELU(),
        logBER=True,
        batch_norm=False,
        dropout=0.42,
    ).to(device)

    learning_rate = 0.0088
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=39)

    #"""
    test_predictor_configuration(
        title="DeepBER Baseline",
        device=device,
        model=model,
        dataloader=pred_dataloader_combined,
        learning_rate=learning_rate,
        batch_size=batch_size_combined,
        criterion=criterion,
        optimizer=optimizer,
        # scheduler=scheduler,
        epochs=240,
        early_stopping=True,
        patience=10,
        training_curves=True,
        predicted_vs_actual=True,
        # error_distribution=True,
        # error_vs_feature=feature_columns,
        # feature_columns=feature_columns
    )
    #"""

    feature_arrays =[
        # [1.3, 2.1, 4.9, 415.0, 1.14, 1.6, 1.15, 51.25, 578.75, 16.5, 1.3 / 2.1, 1.3 * 1.6, 4.9 / 1.3],
        # [1.3, 3.72, 2.25, 388.0, 0.8, 1.28, 0.8, 45.75, 287.75, 63.5, 1.3 / 3.72, 1.3 * 1.28, 2.25 / 1.3],
        [2.3, 2.7, 1.1, 366.0, 0.46, 2.0, 0.55, 31.75, 534.0, 39.0, 2.3 / 2.7, 2.3 * 2.0, 1.1 / 2.3],
    ]

    # Compute standardization parameters from combined training data
    feature_mean = x_array_combined.mean(axis=0)
    feature_std = x_array_combined.std(axis=0)
    feature_std = np.where(feature_std == 0.0, 1.0, feature_std)

    ber_vs_length_test(
        model=model,
        feature_arrays=feature_arrays,
        length_interval=[1500, 4900],
        number_of_points=100,
        feature_columns=feature_columns_combined,
        feature_mean=feature_mean,
        feature_std=feature_std,
        visualization=True,
    )

    """
    run_optuna(
        x_array_combined, 
        y_array_combined_log, 
        feature_columns_combined, 
        gray_area_interval=[-5.5, -2.5], 
        n_trials=800, 
        n_epochs=240, 
        seed=42, 
        study_name="deepber_optuna_combined", 
        cv_folds=3
    )
    """


if __name__ == "__main__":
    main()