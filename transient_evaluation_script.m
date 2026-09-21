addpath('transient/visualization');
addpath('transient/freq2time');
addpath('transient/step_and_prbs_eye');
addpath('transient/pda');
addpath('transient/error_statistical_analysis');
addpath('transient');
addpath('transient/fsv');
addpath('transient/fsv_statistical_analysis');

% Configure Simulation Parameters
start_geom = 1;
max_geoms = 46;
filename_s_params = "out_files/pi_stcnn/touchstone_files_total";
filename_amplitude = ""; %"out_files/amplitude_prediction/export4transient/amplitude_predictions.mat";

run_step_and_prbs_eye = true;
run_pda = false;
show_transient_plots = false;

single_channel = true;
bit_rate = 4e9;

run_fsv = false;
show_fsv_plots = false;

show_statistics_plots = true;

% --- NEW: Quality Check Configuration ---
run_quality_check = false;
quality_threshold = 80; % IEEE 370 score > 80 is 'Good', > 50 is 'Acceptable'
% ----------------------------------------

if filename_amplitude ~= ""
    amplitude_correction_data_all_geoms = load(filename_amplitude, 'Geom_Index', 'V_out_pred', 'V_out_target');    
else
    amplitude_correction_data_all_geoms = [];
end

% Initialize 
step_avg_rmse = 0;
eye_height_avg_rmse = 0;
eye_width_avg_rmse = 0;
eye_height_avg_mape = 0;
eye_width_avg_mape = 0;

pda_avg_eye_height_rmse = 0;
pda_avg_eye_width_rmse = 0;
pda_avg_verdict_error_percentage = 0;
pda_avg_eye_height_mape = 0;
pda_avg_eye_width_mape = 0;

global_prbs_EH_pred = []; global_prbs_EH_act = [];
global_prbs_EW_pred = []; global_prbs_EW_act = [];
global_pda_EH_pred = []; global_pda_EH_act = [];
global_pda_EW_pred = []; global_pda_EW_act = [];
global_pda_Verdict_pred = []; global_pda_Verdict_act = [];
global_ber_pred = []; global_ber_act = [];

% --- NEW: Initialize Quality Check Counters ---
causal_samples_count = 0;
passive_samples_count = 0;
% ----------------------------------------------

if run_fsv
    global_ADMc_geoms = zeros(max_geoms, 6);
    global_FDMc_geoms = zeros(max_geoms, 6);
    global_GDMc_geoms = zeros(max_geoms, 6);

    % Initialize arrays for raw numerical statistics
    global_ADM_all = [];
    global_FDM_all = [];
    global_GDM_all = [];
end

for geom_idx = start_geom:(start_geom + max_geoms - 1)
    if geom_idx == 24
        fprintf('Skipping geometry %d due to known issues with S-parameter data.\n', geom_idx);
        continue;
    end

    geometry_title = sprintf('Geometry %d', geom_idx);

    % Load s-Parameters and amplitude correction data
    filename_preds = string(filename_s_params) + "/preds/geom" + geom_idx + "_pred.s18p";
    filename_actuals = string(filename_s_params) + "/actuals/geom" + geom_idx + "_actual.s18p";

    % --- NEW: Causality and Passivity Check ---
    if run_quality_check
        % The function natively accepts Touchstone file paths.
        % It returns [Causality, Reciprocity, Passivity] metrics.
        % We use '~' to ignore the reciprocity metric for now.
        [cqm_pred, ~, pqm_pred] = ieee370QualityCheckFrequencyDomain(filename_actuals);
        
        % Check if the predictions meet the quality threshold
        if cqm_pred > quality_threshold
            causal_samples_count = causal_samples_count + 1;
        end
        if pqm_pred > quality_threshold
            passive_samples_count = passive_samples_count + 1;
        end
        
        % Print per-sample metrics to the console
        fprintf('%s - Predicted S-Params Quality -> Causality: %.2f%% | Passivity: %.2f%%\n', geometry_title, cqm_pred, pqm_pred);
    end
    % ------------------------------------------
    
    if ~isempty(amplitude_correction_data_all_geoms)
        amplitude_correction_data = struct('V_out_pred', amplitude_correction_data_all_geoms.V_out_pred(geom_idx), ... 
            'V_out_target', amplitude_correction_data_all_geoms.V_out_target(geom_idx));
    else
        amplitude_correction_data = struct();
    end
    
    if run_step_and_prbs_eye
    [prbs_data, step_metrics, eye_metrics, ber_data] = run_transient_evaluation(filename_preds, filename_actuals, amplitude_correction_data, geometry_title, ... 
                                            show_transient_plots, single_channel, bit_rate);
        step_avg_rmse = step_avg_rmse + step_metrics.avg_rmse_main;
        eye_height_avg_rmse = eye_height_avg_rmse + eye_metrics.avg_rmse_eye_height;
        eye_width_avg_rmse = eye_width_avg_rmse + eye_metrics.avg_rmse_eye_width;
        eye_height_avg_mape = eye_height_avg_mape + eye_metrics.avg_mape_eye_height;
        eye_width_avg_mape = eye_width_avg_mape + eye_metrics.avg_mape_eye_width;

        % Filter out NaNs 
        valid_idx = ~isnan(prbs_data.EH_pred);
        global_prbs_EH_pred = [global_prbs_EH_pred, prbs_data.EH_pred(valid_idx)];
        global_prbs_EH_act  = [global_prbs_EH_act,  prbs_data.EH_act(valid_idx)];
        global_prbs_EW_pred = [global_prbs_EW_pred, prbs_data.EW_pred(valid_idx)];
        global_prbs_EW_act  = [global_prbs_EW_act,  prbs_data.EW_act(valid_idx)];

        global_ber_pred = [global_ber_pred, ber_data.pred];
        global_ber_act  = [global_ber_act,  ber_data.act];
    end

    if run_pda
        [pda_data, pda_metrics] = run_pda_evaluation(filename_preds, filename_actuals, amplitude_correction_data, geometry_title, show_transient_plots, ...
                                single_channel, bit_rate);
        pda_avg_eye_height_rmse = pda_avg_eye_height_rmse + pda_metrics.avg_eye_height_rmse;
        pda_avg_eye_width_rmse = pda_avg_eye_width_rmse + pda_metrics.avg_eye_width_rmse;
        pda_avg_verdict_error_percentage = pda_avg_verdict_error_percentage + pda_metrics.verdict_error_percentage;
        pda_avg_eye_height_mape = pda_avg_eye_height_mape + pda_metrics.avg_eh_mape;
        pda_avg_eye_width_mape = pda_avg_eye_width_mape + pda_metrics.avg_ew_mape;

        valid_idx = ~isnan(pda_data.Pass_pred);
        global_pda_EH_pred = [global_pda_EH_pred, pda_data.EH_pred(valid_idx)];
        global_pda_EH_act  = [global_pda_EH_act,  pda_data.EH_act(valid_idx)];
        global_pda_EW_pred = [global_pda_EW_pred, pda_data.EW_pred(valid_idx)];
        global_pda_EW_act  = [global_pda_EW_act,  pda_data.EW_act(valid_idx)];
        global_pda_Verdict_pred = [global_pda_Verdict_pred, pda_data.Pass_pred(valid_idx)];
        global_pda_Verdict_act  = [global_pda_Verdict_act,  pda_data.Pass_act(valid_idx)];
    end

    if run_fsv
        [geom_ADMc, geom_FDMc, geom_GDMc, ADM_mat, FDM_mat, GDM_mat] = run_fsv_evaluation(filename_preds, filename_actuals, geom_idx, show_fsv_plots);

        geom_array_idx = geom_idx - start_geom + 1;
        global_ADMc_geoms(geom_array_idx, :) = geom_ADMc;
        global_FDMc_geoms(geom_array_idx, :) = geom_FDMc;
        global_GDMc_geoms(geom_array_idx, :) = geom_GDMc;

        % Flatten the NxN matrices and append to the global arrays
        global_ADM_all = [global_ADM_all; ADM_mat(:)];
        global_FDM_all = [global_FDM_all; FDM_mat(:)];
        global_GDM_all = [global_GDM_all; GDM_mat(:)];
    end
end

fprintf('\n\n');

% --- NEW: Quality Summary Output ---
if run_quality_check
    causality_percentage = (causal_samples_count / max_geoms) * 100;
    passivity_percentage = (passive_samples_count / max_geoms) * 100;
    
    fprintf('=== S-Parameter Quality Check Summary (Predicted Models) ===\n');
    fprintf('Total Geometries Evaluated: %d\n', max_geoms);
    fprintf('Passive Samples (>%d%% metric): %.2f%%\n', quality_threshold, passivity_percentage);
    fprintf('Causal Samples  (>%d%% metric): %.2f%%\n\n', quality_threshold, causality_percentage);
end
% -----------------------------------

if run_step_and_prbs_eye || run_pda
    run_error_stat_analysis(global_prbs_EH_pred, global_prbs_EH_act, global_prbs_EW_pred, global_prbs_EW_act, global_pda_EH_pred, global_pda_EH_act, global_pda_EW_pred, ... 
                            global_pda_EW_act, global_pda_Verdict_pred, global_pda_Verdict_act, show_statistics_plots);
    
    if ~isempty(global_ber_pred) && ~isempty(global_ber_act)
        run_ber_stat_analysis(global_ber_pred, global_ber_act, show_statistics_plots, false);
    end
end

if run_fsv
    run_fsv_stat_analysis(global_ADM_all, global_FDM_all, global_GDM_all, global_ADMc_geoms, global_FDMc_geoms, global_GDMc_geoms, start_geom, max_geoms);  
end




