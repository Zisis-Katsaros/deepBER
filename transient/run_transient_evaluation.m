function run_transient_evaluation(start_geom, max_geoms, s_param_folder_name, run_step_and_prbs_eye, run_pda, run_fsv, run_tdr, run_quality_check, show_transient_plots, ... 
                                    show_statistics_plots, show_fsv_plots, bit_rate, single_channel, amplitude_correction_filename)
    % Transient evaluation wrapper function
    arguments
        start_geom (1,1) {mustBeInteger, mustBePositive}
        max_geoms (1,1) {mustBeInteger, mustBePositive}
        s_param_folder_name (1,:) char
        run_step_and_prbs_eye (1,1) logical
        run_pda (1,1) logical
        run_fsv (1,1) logical
        run_tdr (1,1) logical
        run_quality_check (1,1) logical
        show_transient_plots (1,1) logical
        show_statistics_plots (1,1) logical
        show_fsv_plots (1,1) logical
        bit_rate (1,1) double {mustBePositive}
        single_channel (1,1) logical = true
        amplitude_correction_filename (1,:) char = ""
    end

    if amplitude_correction_filename ~= ""
        amplitude_correction_data_all_geoms = load(amplitude_correction_filename, 'Geom_Index', 'V_out_pred', 'V_out_target');    
    else
        amplitude_correction_data_all_geoms = [];
    end

    % Initialize metrics and lists for global statistics

    % Step and PRBS Eye
    step_avg_rmse = 0;
    eye_height_avg_rmse = 0;
    eye_width_avg_rmse = 0;
    eye_height_avg_mape = 0;
    eye_width_avg_mape = 0;
    global_prbs_EH_pred = []; global_prbs_EH_act = [];
    global_prbs_EW_pred = []; global_prbs_EW_act = [];
    global_ber_pred = []; global_ber_act = [];

    % PDA
    pda_avg_eye_height_rmse = 0;
    pda_avg_eye_width_rmse = 0;
    pda_avg_verdict_error_percentage = 0;
    pda_avg_eye_height_mape = 0;
    pda_avg_eye_width_mape = 0;
    global_pda_EH_pred = []; global_pda_EH_act = [];
    global_pda_EW_pred = []; global_pda_EW_act = [];
    global_pda_Verdict_pred = []; global_pda_Verdict_act = [];

    % FSV
    global_ADMc_geoms = zeros(max_geoms, 6);
    global_FDMc_geoms = zeros(max_geoms, 6);
    global_GDMc_geoms = zeros(max_geoms, 6);
    global_ADM_all = [];
    global_FDM_all = [];
    global_GDM_all = [];

    % Quality Check
    causality_scores = zeros(max_geoms, 1);
    passivity_scores = zeros(max_geoms, 1);

    for geom_idx = start_geom:(start_geom + max_geoms - 1)
        geometry_title = sprintf('Geometry %d', geom_idx);

        % Load s-Parameters and amplitude correction data
        filename_preds = string(s_param_folder_name) + "/preds/geom" + geom_idx + "_pred.s18p";
        filename_actuals = string(s_param_folder_name) + "/actuals/geom" + geom_idx + "_actual.s18p";
   
        if ~isempty(amplitude_correction_data_all_geoms)
            amplitude_correction_data = struct('V_out_pred', amplitude_correction_data_all_geoms.V_out_pred(geom_idx), ... 
                'V_out_target', amplitude_correction_data_all_geoms.V_out_target(geom_idx));
        else
            amplitude_correction_data = struct();
        end

        if run_step_and_prbs_eye
            [prbs_data, step_metrics, eye_metrics, ber_data] = run_step_prbs_evaluation(filename_preds, filename_actuals, amplitude_correction_data, geometry_title, ... 
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

        if run_tdr
            run_tdr_evaluation(filename_preds, filename_actuals, geometry_title, show_transient_plots);
        end

        if run_quality_check
            % The function natively accepts Touchstone file paths. It returns [Causality, Reciprocity, Passivity] metrics.
            [cqm_pred, ~, pqm_pred] = ieee370QualityCheckFrequencyDomain(filename_preds);
            causality_scores(geom_idx) = cqm_pred;
            passivity_scores(geom_idx) = pqm_pred;
        end
    end

    fprintf('\n\n');
    if run_quality_check
        num_good_causality = sum(causality_scores >= 80);
        num_acceptable_causality = sum(causality_scores >= 50 & causality_scores < 80);
        num_inconcusive_causality = sum(causality_scores >= 20 & causality_scores < 50);
        num_poor_causality = sum(causality_scores < 20);

        num_good_passivity = sum(passivity_scores >= 99.9);
        num_acceptable_passivity = sum(passivity_scores >= 99 & passivity_scores < 99.9);
        num_inconcusive_passivity = sum(passivity_scores >= 80 & passivity_scores < 99.9);
        num_poor_passivity = sum(passivity_scores < 80);
        fprintf('Quality Check:\n');
        fprintf('\tCausality\n');
        fprintf('\t\tGood: %d [%.2f%%]\n', num_good_causality, num_good_causality / length(causality_scores) * 100);
        fprintf('\t\tAcceptable: %d [%.2f%%]\n', num_acceptable_causality, num_acceptable_causality / length(causality_scores) * 100);
        fprintf('\t\tInconclusive: %d [%.2f%%]\n', num_inconcusive_causality, num_inconcusive_causality / length(causality_scores) * 100);
        fprintf('\t\tPoor: %d [%.2f%%]\n', num_poor_causality, num_poor_causality / length(causality_scores) * 100);
        fprintf('\tPassivity\n');
        fprintf('\t\tGood: %d [%.2f%%]\n', num_good_passivity, num_good_passivity / length(passivity_scores) * 100);
        fprintf('\t\tAcceptable: %d [%.2f%%]\n', num_acceptable_passivity, num_acceptable_passivity / length(passivity_scores) * 100);
        fprintf('\t\tInconclusive: %d [%.2f%%]\n', num_inconcusive_passivity, num_inconcusive_passivity / length(passivity_scores) * 100);
        fprintf('\t\tPoor: %d [%.2f%%]\n', num_poor_passivity, num_poor_passivity / length(passivity_scores) * 100);
    end

    if run_step_and_prbs_eye || run_pda
        run_error_stat_analysis(global_prbs_EH_pred, global_prbs_EH_act, global_prbs_EW_pred, global_prbs_EW_act, global_pda_EH_pred, global_pda_EH_act, global_pda_EW_pred, ... 
                                global_pda_EW_act, global_pda_Verdict_pred, global_pda_Verdict_act, show_statistics_plots);
        
        if ~isempty(global_ber_pred) && ~isempty(global_ber_act)
            run_ber_stat_analysis(global_ber_pred, global_ber_act, show_statistics_plots, true);
        end
    end

    if run_fsv
        run_fsv_stat_analysis(global_ADM_all, global_FDM_all, global_GDM_all, global_ADMc_geoms, global_FDMc_geoms, global_GDMc_geoms, start_geom, max_geoms);  
    end
end


    
    
    