from prediction.optuna_tuner import run_optuna

import torch
from torch import nn
from load_set import create_dataloader, load_csv_dataset, create_arrays
from dataset_manipulation import extend_features, exclude_columns
from classification.classifier import DeepBERClassifier
from classification.test_classifier_config import test_classifier_configuration
from prediction.predictor import DeepBERPredictor
from prediction.test_predictor_config import test_predictor_configuration, ber_vs_length_test
from classification.ber_to_class import ber_to_class
import numpy as np
from classification.optuna_tuner import run_optuna_classifier
from sklearn.utils.class_weight import compute_class_weight

def main():
    torch.manual_seed(42)

    # ============================================= Initializing Dataset ============================================= #
    csv_names = [
     ["ber_og_database.csv"],
     ["wrst_case_ber_database1.csv", "wrst_case_ber_database2.csv", "wrst_case_ber_database3.csv"],
     ["prbs_case_database1.csv", "prbs_case_database2.csv"],
     ["prbs_case_database1.csv", "prbs_case_database2.csv", "wrst_case_ber_database1.csv", "wrst_case_ber_database2.csv", "wrst_case_ber_database3.csv" ]
    ]

    target_columns = ["BER", "BER", "BER", "BER"]

    thresholds = [(10**-5.5, 10**-2.5), (10**-5.5, 10**-2.5), (10**-5.5, 10**-2.5), (10**-5.5, 10**-2.5)]

    test_names = ["BER_OG Dataset", "Worst-Case BER Dataset", "PRBS Case BER Dataset", "Combined BER Dataset"]

    test_info_dict = create_arrays(csv_names, target_columns, thresholds, test_names)

    batch_size_dict = {
        "BER_OG Dataset": 32,
        "Worst-Case BER Dataset": 16,
        "PRBS Case BER Dataset": 32,
        "Combined BER Dataset": 16
    }

    dataloader_dict = {}

    for test_name, test_info in test_info_dict.items():
        x_array, y_array, _, _, thresholds, _ = test_info
        batch_size = batch_size_dict[test_name]

        dataloader_dict[test_name] = create_dataloader(
            x_array,
            y_array,
            logBER=True,
            batch_size=batch_size,
            seed=42,
            ber_interval=thresholds,
            standard_scale=True
        )
    
    x_array_combined = test_info_dict["Combined BER Dataset"][0]
    y_classes = test_info_dict["Combined BER Dataset"][3]
    feature_columns_combined  = test_info_dict["Combined BER Dataset"][5]

    classifier_dataloader = create_dataloader(
        x_array_combined,
        y_classes,
        logBER=False,
        batch_size=16,
        seed=42,
        standard_scale=True,
    )


    # =================================================== Classifier ================================================== #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    classifier = DeepBERClassifier(
        input_size=len(feature_columns_combined), 
        num_classes=3, 
        hidden=[128, 256, 32], 
        activation_fn=nn.GELU(), 
        logBER=True, 
        batch_norm=False, 
        dropout=0.08
    )
    lower_thres, upper_thres = -5.5, -2.5

    

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(3),
        y=y_classes,
    )

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0032, weight_decay=0.0013)

    test_classifier_configuration(
        title="MLP Classifier",
        model=classifier,
        dataloader=classifier_dataloader,
        lower_thres=lower_thres,
        upper_thres=upper_thres,
        device=device,
        learning_rate=0.0032,
        criterion=criterion,
        optimizer=optimizer,
        epochs=60,
        early_stopping=True,
        patience=10,
        confusion_matrix=True
    )
    
    """
    run_optuna_classifier(
    x_array_combined,
    y_array_combined,
    feature_columns_combined,
    lower_thres=-5.5,
    upper_thres=-2.5,
    logBER=True,
    n_trials=300,
    seed=42,
    study_name="deepber_classifier_optuna",
    storage=None,
    cv_folds=3,
    mlp_epochs=60,
    mlp_patience=10,
    )
    """

    # =================================================== Predictor =================================================== #
    model = DeepBERPredictor(
        input_size=len(test_info_dict["BER_OG Dataset"][4]),  
        hidden=[128, 32, 48],
        activation_fn=nn.GELU(),
        logBER=True,
        batch_norm=False,
        dropout=0.244,
    ).to(device)

    learning_rate = 0.0058
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=39)

    """
    test_predictor_configuration(
        title="DeepBER Best BER_OG Configuration",
        device=device,
        model=model,
        dataloader=pred_dataloader_ber_og,
        learning_rate=learning_rate,
        batch_size=batch_size_ber_og,
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
    """

    feature_arrays =[
        [1.3, 2.1, 4.9, 415.0, 1.14, 1.6, 1.15, 51.25, 578.75, 16.5, 1.3 / 2.1, 1.3 * 1.6, 4.9 / 1.3],
        [1.3, 3.72, 2.25, 388.0, 0.8, 1.28, 0.8, 45.75, 287.75, 63.5, 1.3 / 3.72, 1.3 * 1.28, 2.25 / 1.3],
        [2.3, 2.7, 1.1, 366.0, 0.46, 2.0, 0.55, 31.75, 534.0, 39.0, 2.3 / 2.7, 2.3 * 2.0, 1.1 / 2.3],
    ]

    # Compute standardization parameters from training data
    feature_mean = test_info_dict["BER_OG Dataset"][0].mean(axis=0)
    feature_std = test_info_dict["BER_OG Dataset"][0].std(axis=0)
    feature_std = np.where(feature_std == 0.0, 1.0, feature_std)

    """
    model.load_state_dict(torch.load("ber_og_best_model.pth", map_location=device))

    ber_vs_length_test(
        model=model,
        feature_arrays=feature_arrays,
        length_interval=[1500, 4900],
        number_of_points=100,
        feature_columns=feature_columns_ber_og,
        feature_mean=feature_mean,
        feature_std=feature_std,
        visualization=True,
        title="BER_OG"
    )
    """

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