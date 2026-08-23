addpath('transient');

start_geom = 1;
max_geoms = 5;
filename_amplitude = "out_files/amplitude_prediction/export4transient/amplitude_predictions.mat";
run_step_and_prbs_eye = true;
run_pda = false;
show_plots = true;
single_channel = true;

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

for geom_idx = start_geom:(start_geom + max_geoms - 1)
    % Load s-Parameters and amplitude correction data
    filename_preds = ['out_files/pi_stcnn/touchstone_files_lo2/preds/geom', num2str(geom_idx), '_pred.s18p']; 
    filename_actuals = ['out_files/pi_stcnn/touchstone_files_lo2/actuals/geom', num2str(geom_idx), '_actual.s18p']; 
    
    if ~isempty(amplitude_correction_data_all_geoms)
        amplitude_correction_data = struct('V_out_pred', amplitude_correction_data_all_geoms.V_out_pred(geom_idx), ... 
            'V_out_target', amplitude_correction_data_all_geoms.V_out_target(geom_idx));
    else
        amplitude_correction_data = [];
    end
    
    if run_step_and_prbs_eye
    [step_metrics, eye_metrics] = run_transient_evaluation(filename_preds, filename_actuals, amplitude_correction_data, sprintf('Geometry %d', geom_idx), show_plots, single_channel);
        step_avg_rmse = step_avg_rmse + step_metrics.avg_rmse_main;
        eye_height_avg_rmse = eye_height_avg_rmse + eye_metrics.avg_rmse_eye_height;
        eye_width_avg_rmse = eye_width_avg_rmse + eye_metrics.avg_rmse_eye_width;
        eye_height_avg_mape = eye_height_avg_mape + eye_metrics.avg_mape_eye_height;
        eye_width_avg_mape = eye_width_avg_mape + eye_metrics.avg_mape_eye_width;
    end

    if run_pda
        pda_metrics = run_pda_evaluation(filename_preds, filename_actuals, amplitude_correction_data, sprintf('Geometry %d', geom_idx), show_plots, single_channel);
        pda_avg_eye_height_rmse = pda_avg_eye_height_rmse + pda_metrics.avg_eye_height_rmse;
        pda_avg_eye_width_rmse = pda_avg_eye_width_rmse + pda_metrics.avg_eye_width_rmse;
        pda_avg_verdict_error_percentage = pda_avg_verdict_error_percentage + pda_metrics.verdict_error_percentage;
        pda_avg_eye_height_mape = pda_avg_eye_height_mape + pda_metrics.avg_eh_mape;
        pda_avg_eye_width_mape = pda_avg_eye_width_mape + pda_metrics.avg_ew_mape;
    end
end

if run_step_and_prbs_eye
    step_avg_rmse = step_avg_rmse / max_geoms;
    eye_height_avg_rmse = eye_height_avg_rmse / max_geoms;
    eye_width_avg_rmse = eye_width_avg_rmse / max_geoms;
    eye_height_avg_mape = eye_height_avg_mape / max_geoms;
    eye_width_avg_mape = eye_width_avg_mape / max_geoms;
    fprintf('Average Step RMSE across %d geometries: %.4f\n', max_geoms, step_avg_rmse);
    fprintf('Average Eye Height RMSE across %d geometries: %.4f\n', max_geoms, eye_height_avg_rmse);
    fprintf('Average Eye Width RMSE across %d geometries: %.4f\n', max_geoms, eye_width_avg_rmse);
    fprintf('Average Eye Height MAPE across %d geometries: %.4f%%\n', max_geoms, eye_height_avg_mape);
    fprintf('Average Eye Width MAPE across %d geometries: %.4f%%\n', max_geoms, eye_width_avg_mape);
end

if run_pda
    pda_avg_eye_height_rmse = pda_avg_eye_height_rmse / max_geoms;
    pda_avg_eye_width_rmse = pda_avg_eye_width_rmse / max_geoms;
    pda_avg_verdict_error_percentage = pda_avg_verdict_error_percentage / max_geoms;
    pda_avg_eye_height_mape = pda_avg_eye_height_mape / max_geoms;
    pda_avg_eye_width_mape = pda_avg_eye_width_mape / max_geoms;
    fprintf('Average PDA Eye Height RMSE across %d geometries: %.4f\n', max_geoms, pda_avg_eye_height_rmse);
    fprintf('Average PDA Eye Width RMSE across %d geometries: %.4f\n', max_geoms, pda_avg_eye_width_rmse);
    fprintf('Average PDA Verdict Error Percentage across %d geometries: %.4f%%\n', max_geoms, pda_avg_verdict_error_percentage);
    fprintf('Average PDA Eye Height MAPE across %d geometries: %.4f%%\n', max_geoms, pda_avg_eye_height_mape);
    fprintf('Average PDA Eye Width MAPE across %d geometries: %.4f%%\n', max_geoms, pda_avg_eye_width_mape);
end
