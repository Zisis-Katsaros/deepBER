function [pda_data, pda_metrics] = run_pda_evaluation(filename_preds, filename_actuals, amplitude_correction_data, title_str, show_plots, single_channel, fs, bit_rate, Vhi)
    %{
    Runs Peak Distortion Analysis (PDA) on predicted and actual S-parameters.
    Identifies the worst-case channel and evaluates pass/fail against the UCIe mask.
    %}
    arguments
        filename_preds (1,1) string
        filename_actuals (1,1) string
        amplitude_correction_data = []
        title_str (1,1) string = "PDA Evaluation"
        show_plots (1,1) logical = true
        single_channel (1,1) logical = false
        fs (1,1) double = 2e12 % 0.5 ps resolution
        bit_rate (1,1) double = 16e9 % 32 GT/s UCIe Standard
        Vhi (1,1) double = 0.625
    end

    Ts = 1/fs;
    
    % UCIe 32 GT/s Specification Mask
    mask_height = 40e-3; % 40 mV
    mask_width = 20e-12; % 20 ps

    fprintf('\n======================================================\n');
    fprintf('[PDA] Starting Peak Distortion Analysis: %s\n', title_str);
    fprintf('======================================================\n');

    worst_port = 0;
    min_global_eye_height = inf;

    % Arrays to hold metrics for reporting
    results = struct('EH_pred_raw', {}, 'EW_pred_raw', {}, 'Pass_pred_raw', {}, ...
                     'EH_pred_adj', {}, 'EW_pred_adj', {}, 'Pass_pred_adj', {}, ...
                     'EH_act', {}, 'EW_act', {}, 'Pass_act', {});

    % Preallocate arrays with NaN to support correct averaging (especially for single_channel)
    num_ports = 9;
    eh_errors = nan(1, num_ports);
    ew_errors = nan(1, num_ports);
    verdict_mismatches = nan(1, num_ports);
    eh_mapes = nan(1, num_ports);
    ew_mapes = nan(1, num_ports);
    pda_data = struct('Pass_pred', nan(1, num_ports), 'Pass_act', nan(1, num_ports), 'EH_pred', nan(1, num_ports), 'EH_act', nan(1, num_ports), 'EW_pred', nan(1, num_ports), ... 
                        'EW_act', nan(1, num_ports));

    if single_channel
        start_port = 5;
        end_port = 5;
    else
        start_port = 1;
        end_port = 9;
    end
    
    for port = start_port:end_port
        fprintf('\n[PDA] Evaluating Port %d...\n', port);
        
        % Map ports (Main, NEXT, FEXT)
        tx = port; rx = tx + 9;
        next1 = tx - 1; next2 = tx + 1;
        fext1 = rx - 1; fext2 = rx + 1;
        
        if port == 1
            next1 = []; fext1 = [];
        elseif port == 9
            next2 = []; fext2 = [];
        end

        % Extract VTF rational models 
        [fit_main_pred, fit_next1_pred, fit_fext1_pred, fit_next2_pred, fit_fext2_pred] = ...
            s_params2vtf_models(filename_preds, tx, rx, next1, fext1, next2, fext2);
            
        [fit_main_act, fit_next1_act, fit_fext1_act, fit_next2_act, fit_fext2_act] = ...
            s_params2vtf_models(filename_actuals, tx, rx, next1, fext1, next2, fext2);

        % Group crosstalk models into cell arrays
        xtalk_fits_pred = {fit_next1_pred, fit_fext1_pred, fit_next2_pred, fit_fext2_pred};
        xtalk_fits_act = {fit_next1_act, fit_fext1_act, fit_next2_act, fit_fext2_act};
        
        % Calculate Alpha Correction (by simulating a 5ns step response for steady state)
        if ~isempty(amplitude_correction_data) && isfield(amplitude_correction_data, 'V_out_pred')
            t_ss = (0:Ts:5e-9)';
            v_in_ss = min(t_ss / 15e-12, 1) * Vhi; % Quick 15ps step
            v_out_ss = timeresp(fit_main_pred, v_in_ss, Ts);
            
            target_amplitude = amplitude_correction_data.V_out_pred * Vhi;
            alpha_correction = target_amplitude / v_out_ss(end);
        else
            alpha_correction = [];
        end

        % Run PDA for Actual and Raw Prediction
        [s1_act, s0_act, metrics_act] = perform_pda(fit_main_act, xtalk_fits_act, Ts, bit_rate, Vhi, mask_height, mask_width, 1.0);
        [s1_pred_raw, s0_pred_raw, metrics_pred_raw] = perform_pda(fit_main_pred, xtalk_fits_pred, Ts, bit_rate, Vhi, mask_height, mask_width, 1.0);

        % Run PDA for Adjusted Prediction (if available)
        if ~isempty(alpha_correction)
            [s1_pred_adj, s0_pred_adj, metrics_pred_adj] = perform_pda(fit_main_pred, xtalk_fits_pred, Ts, bit_rate, Vhi, mask_height, mask_width, alpha_correction);
        else
            s1_pred_adj = []; s0_pred_adj = []; metrics_pred_adj = [];
        end

        % Safe MAPE Calculation for Raw
        eh_mape_raw = safe_mape(metrics_act.eye_height, metrics_pred_raw.eye_height, 1e-4);
        ew_mape_raw = safe_mape(metrics_act.eye_width, metrics_pred_raw.eye_width, 1e-15);

        % Store metrics
        results(port).EH_act = metrics_act.eye_height;
        results(port).EW_act = metrics_act.eye_width;
        results(port).Pass_act = metrics_act.passes_mask;
        
        results(port).EH_pred_raw = metrics_pred_raw.eye_height;
        results(port).EW_pred_raw = metrics_pred_raw.eye_width;
        results(port).Pass_pred_raw = metrics_pred_raw.passes_mask;
        
        if ~isempty(metrics_pred_adj)
            % Safe MAPE Calculation for Adjusted
            eh_mape_adj = safe_mape(metrics_act.eye_height, metrics_pred_adj.eye_height, 1e-4);
            ew_mape_adj = safe_mape(metrics_act.eye_width, metrics_pred_adj.eye_width, 1e-15);

            results(port).EH_pred_adj = metrics_pred_adj.eye_height;
            results(port).EW_pred_adj = metrics_pred_adj.eye_width;
            results(port).Pass_pred_adj = metrics_pred_adj.passes_mask;
            
            % Use adjusted metrics for error calculation
            eh_errors(port) = metrics_pred_adj.eye_height - metrics_act.eye_height;
            ew_errors(port) = metrics_pred_adj.eye_width - metrics_act.eye_width;
            verdict_mismatches(port) = (metrics_pred_adj.passes_mask ~= metrics_act.passes_mask);
            eh_mapes(port) = eh_mape_adj;
            ew_mapes(port) = ew_mape_adj;

            pda_data.Pass_pred(port) = double(metrics_pred_adj.passes_mask);
            pda_data.Pass_act(port)  = double(metrics_act.passes_mask);
            pda_data.EH_pred(port)   = metrics_pred_adj.eye_height;
            pda_data.EH_act(port)    = metrics_act.eye_height;
            pda_data.EW_pred(port)   = metrics_pred_adj.eye_width;
            pda_data.EW_act(port)    = metrics_act.eye_width;

            % Print 3-way Comparison
            fprintf('\t           PREDICTED (RAW) | PREDICTED (ADJ) |   ACTUAL\n');
            fprintf('\tEye Height: %.4f V       |  %.4f V       |   %.4f V \n', metrics_pred_raw.eye_height, metrics_pred_adj.eye_height, metrics_act.eye_height);
            fprintf('\tEye Width:  %.2f ps       |  %.2f ps       |   %.2f ps \n', metrics_pred_raw.eye_width*1e12, metrics_pred_adj.eye_width*1e12, metrics_act.eye_width*1e12);
            fprintf('\tJitter:     %.2f ps       |  %.2f ps       |   %.2f ps \n', metrics_pred_raw.jitter*1e12, metrics_pred_adj.jitter*1e12, metrics_act.jitter*1e12);
            fprintf('\tUCIe Mask:  %-14s |  %-14s |   %-9s \n', string(metrics_pred_raw.passes_mask), string(metrics_pred_adj.passes_mask), string(metrics_act.passes_mask));
            fprintf('\tEH MAPE:    %6.2f %%       |  %6.2f %%       |     - \n', eh_mape_raw, eh_mape_adj);
            fprintf('\tEW MAPE:    %6.2f %%       |  %6.2f %%       |     - \n', ew_mape_raw, ew_mape_adj);
        else
            % Use raw metrics for error calculation
            eh_errors(port) = metrics_pred_raw.eye_height - metrics_act.eye_height;
            ew_errors(port) = metrics_pred_raw.eye_width - metrics_act.eye_width;
            verdict_mismatches(port) = (metrics_pred_raw.passes_mask ~= metrics_act.passes_mask);
            eh_mapes(port) = eh_mape_raw;
            ew_mapes(port) = ew_mape_raw;

            pda_data.Pass_pred(port) = double(metrics_pred_raw.passes_mask);
            pda_data.Pass_act(port)  = double(metrics_act.passes_mask);
            pda_data.EH_pred(port)   = metrics_pred_raw.eye_height;
            pda_data.EH_act(port)    = metrics_act.eye_height;
            pda_data.EW_pred(port)   = metrics_pred_raw.eye_width;
            pda_data.EW_act(port)    = metrics_act.eye_width;
            
            % Print 2-way Comparison
            fprintf('\t           PREDICTED |   ACTUAL\n');
            fprintf('\tEye Height: %.4f V   |   %.4f V \n', metrics_pred_raw.eye_height, metrics_act.eye_height);
            fprintf('\tEye Width:  %.2f ps   |   %.2f ps \n', metrics_pred_raw.eye_width*1e12, metrics_act.eye_width*1e12);
            fprintf('\tJitter:     %.2f ps   |   %.2f ps \n', metrics_pred_raw.jitter*1e12, metrics_act.jitter*1e12);
            fprintf('\tUCIe Mask:  %-9s |   %-9s \n', string(metrics_pred_raw.passes_mask), string(metrics_act.passes_mask));
            fprintf('\tEH MAPE:    %6.2f %%  |     - \n', eh_mape_raw);
            fprintf('\tEW MAPE:    %6.2f %%  |     - \n', ew_mape_raw);
        end
        
        if show_plots
            plot_title_str = sprintf('%s - PDA Worst-Case Eye (Port %d)', title_str, port);
            plot_pda_eye(Ts, s1_pred_raw, s0_pred_raw, metrics_pred_raw, s1_act, s0_act, metrics_act, ...
                mask_height, mask_width, plot_title_str, s1_pred_adj, s0_pred_adj, metrics_pred_adj);
        end
        
        % Track Worst Channel (Based on Actual Eye Height)
        if metrics_act.eye_height < min_global_eye_height
            min_global_eye_height = metrics_act.eye_height;
            worst_port = port;
        end
    end

    % Calculate final metrics gracefully skipping NaNs 
    pda_metrics = struct();
    pda_metrics.avg_eye_height_rmse = sqrt(mean(eh_errors.^2, 'omitnan'));
    pda_metrics.avg_eye_width_rmse = sqrt(mean(ew_errors.^2, 'omitnan'));
    pda_metrics.verdict_error_percentage = mean(double(verdict_mismatches), 'omitnan') * 100;
    pda_metrics.avg_eh_mape = mean(eh_mapes, 'omitnan');
    pda_metrics.avg_ew_mape = mean(ew_mapes, 'omitnan');

    fprintf('\n======================================================\n');
    fprintf('               WORST CHANNEL ANALYSIS\n');
    fprintf('======================================================\n');
    fprintf('>> The worst-case bottleneck is PORT %d.\n', worst_port);
    fprintf('>> Actual Eye Height: %.4f V\n', results(worst_port).EH_act);
    fprintf('>> Actual Eye Width:  %.2f ps\n', results(worst_port).EW_act*1e12);
    
    if ~isempty(amplitude_correction_data)
        fprintf('>> Predicted Verdict (Raw): %s | Predicted (Adj): %s | Actual: %s\n', ...
                string(results(worst_port).Pass_pred_raw), string(results(worst_port).Pass_pred_adj), string(results(worst_port).Pass_act));
    else
        fprintf('>> Predicted Verdict: %s | Actual Verdict: %s\n', ...
                string(results(worst_port).Pass_pred_raw), string(results(worst_port).Pass_act));
    end

    fprintf('\n======================================================\n');
    fprintf('                 OVERALL PDA METRICS\n');
    fprintf('======================================================\n');
    fprintf('>> Average Eye Height RMSE: %.4f V\n', pda_metrics.avg_eye_height_rmse);
    fprintf('>> Average Eye Width RMSE:  %.4e s\n', pda_metrics.avg_eye_width_rmse);
    fprintf('>> Average Eye Height MAPE: %.2f%%\n', pda_metrics.avg_eh_mape);
    fprintf('>> Average Eye Width MAPE:  %.2f%%\n', pda_metrics.avg_ew_mape);
    fprintf('>> Verdict Error Rate:      %.2f%%\n', pda_metrics.verdict_error_percentage);
end