addpath('transient');

start_geom = 2;
max_geoms = 1;
filename_amplitude = "out_files/amplitude_prediction/export4transient/amplitude_predictions.mat";

if filename_amplitude ~= ""
    amplitude_correction_data_all_geoms = load(filename_amplitude, 'Geom_Index', 'V_out_pred', 'V_out_target');    
else
    amplitude_correction_data_all_geoms = [];
end

for geom_idx = start_geom:(start_geom + max_geoms - 1)
    filename_preds = ['out_files/pi_stcnn/touchstone_files_lo2/preds/geom', num2str(geom_idx), '_pred.s18p']; 
    filename_actuals = ['out_files/pi_stcnn/touchstone_files_lo2/actuals/geom', num2str(geom_idx), '_actual.s18p']; 
    
    if ~isempty(amplitude_correction_data_all_geoms)
        amplitude_correction_data = struct('V_out_pred', amplitude_correction_data_all_geoms.V_out_pred(geom_idx), ... 
            'V_out_target', amplitude_correction_data_all_geoms.V_out_target(geom_idx));
    else
        amplitude_correction_data = [];
    end
        
    % run_transient_evaluation(filename_preds, filename_actuals, amplitude_correction_data, sprintf('Geometry %d', geom_idx));
    run_pda_evaluation(filename_preds, filename_actuals, amplitude_correction_data, sprintf('Geometry %d', geom_idx));
end