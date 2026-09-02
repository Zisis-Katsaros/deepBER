import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from prediction.pi_stcnn_optuna import run_pi_stcnn_optuna
from load_set import organize_dataset_for_pi_stcnn, create_param_dataloader
from prediction.predictor import PI_STCNN
from prediction.l_freq_loss import l_freq_loss
from prediction.test_predictor_config import test_predictor_configuration_pistcnn
import numpy as np
from export_files_for_transient import export_files_for_transient, convert_stcnn_outputs_to_dicts
from dataset_splitting import split_dataset
from prediction.param_pred_optuna_helpers import eval_pistcnn_study
from prediction.parameter_computations import combine_shielded_and_unshielded_portions


# ============================================= Initialization ============================================= #
seed = 42
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

separate_portions = True

# Dataloader Hyperparameters
weight_type = "balanced"

# Predictor Hyperparameters
mlp_hidden = [512, 512, 512, 512]
mlp_dropout = 0.0
tcnn_layer_params=[
            [196, 16, 1],  # [out_channels, kernel_size, stride]
            [196, 4, 2],
            [196, 4, 2],
            [196, 4, 2],
            [196, 4, 2],
            [196, 4, 2],
            [196, 4, 2]
        ]
M=1.5
varience_min=0.1
layer_norm=True

# Learning Hyperparameters
criterion = l_freq_loss().to(device)
learning_rate = 0.001
weight_decay = 0.0 # 5.1635e-05
scheduler_patience = 50
epochs = 3000
patience = 300

max_figures = 3

# ============================================= Training and Testing ============================================= #
if not separate_portions:
    pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict_total.pt", weights_only=False)

    x_array = pred_arrays_dict["x_array"].astype(np.float32)
    s_dict = pred_arrays_dict["s_dict"]
    feature_columns = pred_arrays_dict["feature_columns"]

    x_array, feature_columns, y_array = organize_dataset_for_pi_stcnn(x_array, s_dict, feature_columns)

    dataloader, x_scale_params, y_scale_params, y_weights, *_ = create_param_dataloader(
                        x_array,
                        y_array,
                        batch_size=16,
                        seed=42,
                        standard_scale=(True, False),  # (scale_features, scale_labels)
                        split_method="lhs",
                        weight_type=weight_type
                        )

    _, num_channels_times2, num_freqs = y_array.shape
    predictor = PI_STCNN(
        input_size=len(feature_columns),
        mlp_hidden=mlp_hidden,
        mlp_activation_fn=nn.ELU(),
        mlp_dropout=mlp_dropout,
        tcnn_layer_params=tcnn_layer_params,
        tcnn_activation_fn=nn.ELU(),
        output_size=num_channels_times2 // 2,
        num_ports=18,
        N=num_freqs,
        M=M,
        K=2,
        varience_min=varience_min,
        layer_norm=layer_norm,
    ).to(device)

    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=scheduler_patience) # torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5) 

    test_preds, test_labels = test_predictor_configuration_pistcnn(
        title=f"S-Parameters Prediction with PI-STCNN",
        device=device,
        model=predictor,
        dataloader=dataloader,
        learning_rate=learning_rate,
        batch_size=128,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        phase1_patience=50,
        early_stopping=True,
        patience=patience,
        y_scale_params=y_scale_params,
        training_curves=True,
        predicted_vs_actual=True,
        test_out_dir = f"out_files/pi_stcnn/total",
        close_figures=True,
        max_figures=max_figures,
        max_time_hours=5.5
        )
    
    labels_dict_list, preds_dict_list = convert_stcnn_outputs_to_dicts(test_targets=test_labels, test_preds=test_preds, num_ports=18)
    num_geometries = test_preds.shape[0]

else:
    pred_arrays_dict_shielded = torch.load("csv_files/s_params/pt/pred_arrays_dict_shielded.pt", weights_only=False)
    pred_arrays_dict_unshielded = torch.load("csv_files/s_params/pt/pred_arrays_dict_unshielded.pt", weights_only=False)

    x_array = pred_arrays_dict_shielded["x_array"].astype(np.float32)
    s_dict = pred_arrays_dict_shielded["s_dict"]
    feature_columns = pred_arrays_dict_shielded["feature_columns"]

    x_array, feature_columns, y_array = organize_dataset_for_pi_stcnn(x_array, s_dict, feature_columns)

    dataloader, x_scale_params, y_scale_params, y_weights, split_idx = create_param_dataloader(
                        x_array,
                        y_array,
                        batch_size=16,
                        seed=42,
                        standard_scale=(True, False),  # (scale_features, scale_labels)
                        split_method="lhs",
                        weight_type=weight_type
                        )

    _, num_channels_times2, num_freqs = y_array.shape
    predictor_shielded = PI_STCNN(
        input_size=len(feature_columns),
        mlp_hidden=mlp_hidden,
        mlp_activation_fn=nn.ELU(),
        mlp_dropout=mlp_dropout,
        tcnn_layer_params=tcnn_layer_params,
        tcnn_activation_fn=nn.ELU(),
        output_size=num_channels_times2 // 2,
        num_ports=18,
        N=num_freqs,
        M=M,
        K=2,
        varience_min=varience_min,
        layer_norm=layer_norm,
    ).to(device)

    optimizer_shielded = torch.optim.Adam(predictor_shielded.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler_shielded = ReduceLROnPlateau(optimizer_shielded, mode='min', factor=0.5, patience=scheduler_patience) # torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5) 

    test_preds_shielded, test_labels_shielded = test_predictor_configuration_pistcnn(
        title=f"S-Parameters Prediction with PI-STCNN Shielded Portion",
        device=device,
        model=predictor_shielded,
        dataloader=dataloader,
        learning_rate=learning_rate,
        batch_size=128,
        criterion=criterion,
        optimizer=optimizer_shielded,
        scheduler=scheduler_shielded,
        epochs=epochs,
        phase1_patience=50,
        early_stopping=True,
        patience=patience,
        y_scale_params=y_scale_params,
        training_curves=True,
        predicted_vs_actual=True,
        test_out_dir = f"out_files/pi_stcnn/separate/shielded",
        close_figures=True,
        max_figures=max_figures,
        max_time_hours=5.5
        )

    x_array = pred_arrays_dict_unshielded["x_array"].astype(np.float32)
    s_dict = pred_arrays_dict_unshielded["s_dict"]
    feature_columns = pred_arrays_dict_unshielded["feature_columns"]

    x_array, feature_columns, y_array = organize_dataset_for_pi_stcnn(x_array, s_dict, feature_columns)

    dataloader, x_scale_params, y_scale_params, y_weights, _ = create_param_dataloader(
                        x_array,
                        y_array,
                        batch_size=16,
                        seed=42,
                        standard_scale=(True, False),  # (scale_features, scale_labels)
                        split_method="lhs",
                        weight_type=weight_type,
                        split_idx=split_idx
                        )

    _, num_channels_times2, num_freqs = y_array.shape
    predictor_unshielded = PI_STCNN(
        input_size=len(feature_columns),
        mlp_hidden=mlp_hidden,
        mlp_activation_fn=nn.ELU(),
        mlp_dropout=mlp_dropout,
        tcnn_layer_params=tcnn_layer_params,
        tcnn_activation_fn=nn.ELU(),
        output_size=num_channels_times2 // 2,
        num_ports=18,
        N=num_freqs,
        M=M,
        K=2,
        varience_min=varience_min,
        layer_norm=layer_norm,
    ).to(device)

    optimizer_unshielded = torch.optim.Adam(predictor_unshielded.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler_unshielded = ReduceLROnPlateau(optimizer_unshielded, mode='min', factor=0.5, patience=scheduler_patience) # torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    test_preds_unshielded, test_labels_unshielded = test_predictor_configuration_pistcnn(
            title=f"S-Parameters Prediction with PI-STCNN Unshielded Portion",
            device=device,
            model=predictor_unshielded,
            dataloader=dataloader,
            learning_rate=learning_rate,
            batch_size=16,
            criterion=criterion,
            optimizer=optimizer_unshielded,
            scheduler=scheduler_unshielded,
            epochs=epochs,
            phase1_patience=50,
            early_stopping=True,
            patience=patience,
            y_scale_params=y_scale_params,
            training_curves=True,
            predicted_vs_actual=True,
            test_out_dir = f"out_files/pi_stcnn/separate/unshielded",
            close_figures=True,
            max_figures=max_figures,
            max_time_hours=5.5
            )

    # Combine shielded and unshielded portions into total S-parameter dictionaries lists for labels and predictions
    labels_dict_list_shielded, preds_dict_list_shielded = convert_stcnn_outputs_to_dicts(test_targets=test_labels_shielded, test_preds=test_preds_shielded, num_ports=18)
    labels_dict_list_unshielded, preds_dict_list_unshielded = convert_stcnn_outputs_to_dicts(test_targets=test_labels_unshielded, test_preds=test_preds_unshielded, num_ports=18)

    labels_dict_list_total, preds_dict_list_total = [], []
    num_geometries = test_preds_shielded.shape[0]
    for geom_idx in range(num_geometries):
        labels_s_dict_shielded = labels_dict_list_shielded[geom_idx]
        labels_s_dict_unshielded = labels_dict_list_unshielded[geom_idx]

        labels_s_dict_total = combine_shielded_and_unshielded_portions(labels_s_dict_shielded, labels_s_dict_unshielded)
        labels_dict_list_total.append(labels_s_dict_total)

        preds_s_dict_shielded = preds_dict_list_shielded[geom_idx]
        preds_s_dict_unshielded = preds_dict_list_unshielded[geom_idx]

        preds_s_dict_total = combine_shielded_and_unshielded_portions(preds_s_dict_shielded, preds_s_dict_unshielded)
        preds_dict_list_total.append(preds_s_dict_total)
        
# ============================================= Export for Transient ============================================= #
freq_array = np.linspace(0, 30, 601)
freq_arrays_per_geom = [freq_array for _ in range(num_geometries)]

export_files_for_transient(
    geometries=x_array,  # The unique geometries returned by organize_dataset_for_pi_stcnn
    feature_names=feature_columns,
    labels_dict_per_geom=labels_dict_list if not separate_portions else labels_dict_list_total,
    preds_dict_per_geom=preds_dict_list if not separate_portions else preds_dict_list_total,
    freq_arrays_per_geom=freq_arrays_per_geom,
    save_dir="out_files/pi_stcnn/total/touchstone_files" if not separate_portions else "out_files/pi_stcnn/separate/touchstone_files"
)
# """