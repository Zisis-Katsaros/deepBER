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

run_step_and_prbs_eye = false;
run_pda = false;
run_fsv = false;
run_tdr = false;
run_quality_check = true;

show_transient_plots = true;
show_statistics_plots = true;
show_fsv_plots = true;

single_channel = true;
bit_rate = 16e9;

run_transient_evaluation(start_geom, max_geoms, filename_s_params, run_step_and_prbs_eye, run_pda, run_fsv, run_tdr, run_quality_check, show_transient_plots, ...
                        show_statistics_plots, show_fsv_plots, bit_rate, single_channel, filename_amplitude);





