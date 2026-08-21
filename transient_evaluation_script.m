addpath('transient');

start_geom = 1;
max_geoms = 1;
filename_amplitude = "out_files/amplitude_prediction/export4transient/amplitude_predictions.mat";
run_step_and_prbs_eye = true;
run_pda = false;

if filename_amplitude ~= ""
    amplitude_correction_data_all_geoms = load(filename_amplitude, 'Geom_Index', 'V_out_pred', 'V_out_target');    
else
    amplitude_correction_data_all_geoms = [];
end

% Initialize 
step_avg_rmse = 0;
eye_avg_rmse = 0;
pda_avg_eye_height_rmse = 0;
pda_avg_eye_width_rmse = 0;
pda_avg_verdict_error_percentage = 0;
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
    [step_metrics, eye_metrics] = run_transient_evaluation(filename_preds, filename_actuals, amplitude_correction_data, sprintf('Geometry %d', geom_idx));
        step_avg_rmse = step_avg_rmse + step_metrics.rmse;
        eye_avg_rmse = eye_avg_rmse + eye_metrics.rmse;
    end

    if run_pda
        pda_metrics = run_pda_evaluation(filename_preds, filename_actuals, amplitude_correction_data, sprintf('Geometry %d', geom_idx));
        pda_avg_eye_height_rmse = pda_avg_eye_height_rmse + pda_metrics.avg_eye_height_rmse;
        pda_avg_eye_width_rmse = pda_avg_eye_width_rmse + pda_metrics.avg_eye_width_rmse;
        pda_avg_verdict_error_percentage = pda_avg_verdict_error_percentage + pda_metrics.verdict_error_percentage;
    end
end

if run_step_and_prbs_eye
    step_avg_rmse = step_avg_rmse / max_geoms;
    eye_avg_rmse = eye_avg_rmse / max_geoms;
    printf('Average Step RMSE across %d geometries: %.4f\n', max_geoms, step_avg_rmse);
    printf('Average Eye RMSE across %d geometries: %.4f\n', max_geoms, eye_avg_rmse);
end

if run_pda
    pda_avg_eye_height_rmse = pda_avg_eye_height_rmse / max_geoms;
    pda_avg_eye_width_rmse = pda_avg_eye_width_rmse / max_geoms;
    pda_avg_verdict_error_percentage = pda_avg_verdict_error_percentage / max_geoms;
    printf('Average PDA Eye Height RMSE across %d geometries: %.4f\n', max_geoms, pda_avg_eye_height_rmse);
    printf('Average PDA Eye Width RMSE across %d geometries: %.4f\n', max_geoms, pda_avg_eye_width_rmse);
    printf('Average PDA Verdict Error Percentage across %d geometries: %.4f%%\n', max_geoms, pda_avg_verdict_error_percentage);
end
