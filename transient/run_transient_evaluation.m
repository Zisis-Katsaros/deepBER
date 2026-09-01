function [prbs_data, step_metrics, eye_metrics] = run_transient_evaluation(filename_preds, filename_actuals, amplitude_correction_data, title, show_plots, single_channel, apply_worst_case_xtalk, fs, t_step, rise_time, delay, Vhi, ...
        num_bits, bit_rate, precision)
    arguments
        filename_preds (1,1) string
        filename_actuals (1,1) string
        amplitude_correction_data (1,1) struct = []
        title (1,1) string = "Transient Evaluation"
        show_plots (1,1) logical = true
        single_channel (1,1) logical = false
        apply_worst_case_xtalk (1,1) logical = false
        fs (1,1) double = 1e12
        t_step (1,1) double = 2e-9
        rise_time (1,1) double {mustBePositive} = 15e-12;
        delay (1,1) double {mustBeNonnegative} = 100e-12; 
        Vhi (1,1) double {mustBePositive} = 0.625; 
        num_bits (1,1) double {mustBeInteger, mustBePositive} = 1000;
        bit_rate (1,1) double {mustBePositive} = 32e9; 
        precision = -40;
    end

    if ~isempty(amplitude_correction_data) && isfield(amplitude_correction_data, 'V_out_pred')
        V_out_pred_val = amplitude_correction_data.V_out_pred * Vhi; 
        V_out_target_val = amplitude_correction_data.V_out_target * Vhi; 
    else
        V_out_pred_val = [];
        V_out_target_val = [];
    end

    samples_per_bit = round(fs / bit_rate);
    fs = samples_per_bit * bit_rate; 

    Ts = 1/fs; 
    t = (0:Ts:t_step)';

    V_in_step = lo2hi_step_stimulus(t, rise_time, delay, Vhi);
    V_in_prbs = prbs_stimulus(num_bits, bit_rate, rise_time, Ts);
    
    if single_channel
        start_port = 5; end_port = 5;
    else
        start_port = 1; end_port = 9;
    end

    num_ports = 9;
    rmse_step = struct('main', nan(1, num_ports), 'next1', nan(1, num_ports), 'fext1', nan(1, num_ports), 'next2', nan(1, num_ports), 'fext2', nan(1, num_ports));
    rmse_eye = struct('rt', nan(1, num_ports), 'ft', nan(1, num_ports), 'height', nan(1, num_ports), 'width', nan(1, num_ports), 'jitter', nan(1, num_ports), 'amp', nan(1, num_ports));
    mape_eye = struct('height', nan(1, num_ports), 'width', nan(1, num_ports));
    prbs_data = struct('EH_pred', nan(1, num_ports), 'EH_act', nan(1, num_ports), 'EW_pred', nan(1, num_ports), 'EW_act', nan(1, num_ports));

    fprintf("[transient evaluation] Beginning %s\n", title);
    for port = start_port:end_port
        fprintf("[transient evaluation] \tEvaluating port %d...\n", port);
        
        tx = port; rx = tx + 9;
        next1 = tx - 1; next2 = tx + 1;
        fext1 = rx - 1; fext2 = rx + 1;

        if port == 1
            next1 = tx; fext1 = rx;      
        elseif port == 9
            next2 = tx; fext2 = rx;
        end
        
        % S-parameters to Impulse Response Conversion
        [fit_main_pred, fit_next1_pred, fit_fext1_pred, fit_next2_pred, fit_fext2_pred] = s_params2impulse_response(filename_preds, tx, rx, next1, fext1, next2, fext2, precision);
        [fit_main_actual, fit_next1_actual, fit_fext1_actual, fit_next2_actual, fit_fext2_actual] = s_params2impulse_response(filename_actuals, tx, rx, next1, fext1, next2, fext2, precision);
        
        % timeresp(model, V_in, Ts);
        % Predicted step response
        V_out_main_step_pred = timeresp(fit_main_pred, V_in_step, Ts);
        V_out_next1_step_pred = timeresp(fit_next1_pred, V_in_step, Ts);
        V_out_fext1_step_pred = timeresp(fit_fext1_pred, V_in_step, Ts);
        V_out_next2_step_pred = timeresp(fit_next2_pred, V_in_step, Ts);
        V_out_fext2_step_pred = timeresp(fit_fext2_pred, V_in_step, Ts);
        
        if ~isempty(V_out_pred_val)
            corr_factor = V_out_pred_val / V_out_main_step_pred(end);
            V_out_main_step_pred_adj = V_out_main_step_pred * corr_factor;
            eval_step_pred = V_out_main_step_pred_adj;
        else
            V_out_main_step_pred_adj = [];
            eval_step_pred = V_out_main_step_pred;
        end

        % Actual step response
        V_out_main_step_actual = timeresp(fit_main_actual, V_in_step, Ts);
        V_out_next1_step_actual = timeresp(fit_next1_actual, V_in_step, Ts);
        V_out_fext1_step_actual = timeresp(fit_fext1_actual, V_in_step, Ts);
        V_out_next2_step_actual = timeresp(fit_next2_actual, V_in_step, Ts);
        V_out_fext2_step_actual = timeresp(fit_fext2_actual, V_in_step, Ts);

        if port == 1
            V_out_next1_step_pred = []; V_out_fext1_step_pred = [];
            V_out_next1_step_actual = []; V_out_fext1_step_actual = [];
        elseif port == 9
            V_out_next2_step_pred = []; V_out_fext2_step_pred = [];
            V_out_next2_step_actual = []; V_out_fext2_step_actual = [];
        end

        rmse_step.main(port) = rmse(eval_step_pred, V_out_main_step_actual);
        if port ~= 1
            rmse_step.next1(port) = rmse(V_out_next1_step_pred, V_out_next1_step_actual);
            rmse_step.fext1(port) = rmse(V_out_fext1_step_pred, V_out_fext1_step_actual);
        end
        if port ~= 9
            rmse_step.next2(port) = rmse(V_out_next2_step_pred, V_out_next2_step_actual);
            rmse_step.fext2(port) = rmse(V_out_fext2_step_pred, V_out_fext2_step_actual);
        end

        fprintf("[transient evaluation] \t>> 0->1 Step stimulus:\n");
        fprintf("[transient evaluation] \t- RMSE (Main): %.4f V\n", rmse_step.main(port));
        if port ~= 1
            fprintf("[transient evaluation] \t- RMSE (NEXT1): %.4f V\n", rmse_step.next1(port));
            fprintf("[transient evaluation] \t- RMSE (FEXT1): %.4f V\n", rmse_step.fext1(port));
        end 
        if port ~= 9
            fprintf("[transient evaluation] \t- RMSE (NEXT2): %.4f V\n", rmse_step.next2(port));
            fprintf("[transient evaluation] \t- RMSE (FEXT2): %.4f V\n", rmse_step.fext2(port));
        end
        
        if show_plots
            plot_step_response_pred_vs_act(t, V_in_step, V_out_main_step_pred, V_out_main_step_actual, ...
                V_out_next1_step_pred, V_out_next1_step_actual, V_out_fext1_step_pred, V_out_fext1_step_actual, ...
                V_out_next2_step_pred, V_out_next2_step_actual, V_out_fext2_step_pred, V_out_fext2_step_actual, ...
                sprintf('%s - Step Response Prediction Vs Actual (Port %d)', title, port), V_out_main_step_pred_adj, V_out_target_val);
        end

        % PRBS responses
        if ~apply_worst_case_xtalk
            V_out_main_prbs_pred = timeresp(fit_main_pred, V_in_prbs, Ts);
            V_out_main_prbs_actual = timeresp(fit_main_actual, V_in_prbs, Ts);
        else
            % Gather available crosstalk channels based on port location
            xtalk_fits_pred = {};
            xtalk_fits_actual = {};
            
            if port ~= 1
                xtalk_fits_pred(end+1:end+2) = {fit_next1_pred, fit_fext1_pred};
                xtalk_fits_actual(end+1:end+2) = {fit_next1_actual, fit_fext1_actual};
            end
            if port ~= 9
                xtalk_fits_pred(end+1:end+2) = {fit_next2_pred, fit_fext2_pred};
                xtalk_fits_actual(end+1:end+2) = {fit_next2_actual, fit_fext2_actual};
            end
            % Apply worst-case dynamic crosstalk
            V_out_main_prbs_pred = apply_xtalk(fit_main_pred, xtalk_fits_pred, V_in_prbs, Vhi, Ts);
            V_out_main_prbs_actual = apply_xtalk(fit_main_actual, xtalk_fits_actual, V_in_prbs, Vhi, Ts);
        end

        if ~isempty(V_out_pred_val)
            V_out_main_prbs_pred_adj = V_out_main_prbs_pred * corr_factor;
            eval_prbs_pred = V_out_main_prbs_pred_adj;
        else
            V_out_main_prbs_pred_adj = [];
            eval_prbs_pred = V_out_main_prbs_pred;
        end

        % === DYNAMIC ALIGNMENT BLOCK ===
        settle_bits = 10;
        
        % Force valid_bits to be an even number to allow reshaping into 2-UI columns
        valid_bits = floor((num_bits - settle_bits - 1) / 2) * 2; 
        
        valid_samples = valid_bits * samples_per_bit; 
        settle_idx = settle_bits * samples_per_bit;

        % Check variance across columns temporarily to find the true first crossing point
        temp_matrix = reshape(V_out_main_prbs_actual(settle_idx + 1 : settle_idx + valid_samples), samples_per_bit * 2, []);
        v_var = var(temp_matrix, 0, 2);
        [~, actual_cross_idx] = min(v_var(1:samples_per_bit));
        
        % Calculate shift required to perfectly align crossing to exactly 0.5 UI
        target_cross_idx = round(samples_per_bit / 2);
        idx_offset = actual_cross_idx - target_cross_idx;

        % Shift the linear read-window. This naturally absorbs the delay and centers the eye geometry natively!
        aligned_start = settle_idx + 1 + idx_offset;
        aligned_end = aligned_start + valid_samples - 1;

        eye_matrix_Vout_pred = reshape(V_out_main_prbs_pred(aligned_start:aligned_end), samples_per_bit * 2, []);
        eye_matrix_Vout_actual = reshape(V_out_main_prbs_actual(aligned_start:aligned_end), samples_per_bit * 2, []);
        
        if ~isempty(V_out_main_prbs_pred_adj)
            eye_matrix_Vout_pred_adj = reshape(V_out_main_prbs_pred_adj(aligned_start:aligned_end), samples_per_bit * 2, []);
            eval_eye_matrix = eye_matrix_Vout_pred_adj;
        else
            eye_matrix_Vout_pred_adj = [];
            eval_eye_matrix = eye_matrix_Vout_pred;
        end
        % ==============================
        
        [rmse_eye.rt(port), rmse_eye.ft(port), rmse_eye.height(port), rmse_eye.jitter(port), rmse_eye.amp(port), rmse_eye.width(port), mape_eye.height(port), mape_eye.width(port), ...
         prbs_data.EH_pred(port), prbs_data.EH_act(port), prbs_data.EW_pred(port), prbs_data.EW_act(port)] = ...
            eye_metrics_pred_vs_act(eval_prbs_pred, V_out_main_prbs_actual, eval_eye_matrix, eye_matrix_Vout_actual, fs, bit_rate);
        
        fprintf("[transient evaluation] \t>> PRBS stimulus:\n");
        fprintf("[transient evaluation] \t- RMSE Rise Time: %.4f s\n", rmse_eye.rt(port));
        fprintf("[transient evaluation] \t- RMSE Fall Time: %.4f s\n", rmse_eye.ft(port));
        fprintf("[transient evaluation] \t- RMSE Eye Height: %.4f V\n", rmse_eye.height(port));
        fprintf("[transient evaluation] \t- RMSE Eye Width: %.4f s\n", rmse_eye.width(port));
        fprintf("[transient evaluation] \t- RMSE Eye Jitter: %.4f s\n", rmse_eye.jitter(port));
        fprintf("[transient evaluation] \t- RMSE Eye Amplitude: %.4f V\n", rmse_eye.amp(port));
        fprintf("[transient evaluation] \t- MAPE Eye Height: %.2f%%\n", mape_eye.height(port));
        fprintf("[transient evaluation] \t- MAPE Eye Width: %.2f%%\n", mape_eye.width(port));
    
        if show_plots
            % Isolate a 1.5 UI window explicitly (from 0.25 UI to 1.75 UI). 
            % Since we dynamically aligned the matrix above, the eye will be mathematically fixed dead-center.
            plot_start = round(0.25 * samples_per_bit) + 1;
            plot_end = round(1.75 * samples_per_bit);
            plot_idx = plot_start:plot_end;
            
            t_eye_plot = linspace(0.25, 1.75, length(plot_idx));
            
            if isempty(eye_matrix_Vout_pred_adj)
                eval_plot_adj = [];
            else
                eval_plot_adj = eye_matrix_Vout_pred_adj(plot_idx, :);
            end

            plot_eye_pred_vs_act(t_eye_plot, eye_matrix_Vout_pred(plot_idx, :), ...
                eye_matrix_Vout_actual(plot_idx, :), ...
                sprintf('%s - Eye Diagram Prediction Vs Actual (Port %d)', title, port), ...
                eval_plot_adj); 
        end   
    end

    step_metrics = struct( ...
        'avg_rmse_main', mean(rmse_step.main, 'omitnan'), 'min_rmse_main', min(rmse_step.main, [], 'omitnan'), 'max_rmse_main', max(rmse_step.main, [], 'omitnan'), ...
        'avg_rmse_next1', mean(rmse_step.next1, 'omitnan'), 'min_rmse_next1', min(rmse_step.next1, [], 'omitnan'), 'max_rmse_next1', max(rmse_step.next1, [], 'omitnan'), ...
        'avg_rmse_fext1', mean(rmse_step.fext1, 'omitnan'), 'min_rmse_fext1', min(rmse_step.fext1, [], 'omitnan'), 'max_rmse_fext1', max(rmse_step.fext1, [], 'omitnan'), ...
        'avg_rmse_next2', mean(rmse_step.next2, 'omitnan'), 'min_rmse_next2', min(rmse_step.next2, [], 'omitnan'), 'max_rmse_next2', max(rmse_step.next2, [], 'omitnan'), ...
        'avg_rmse_fext2', mean(rmse_step.fext2, 'omitnan'), 'min_rmse_fext2', min(rmse_step.fext2, [], 'omitnan'), 'max_rmse_fext2', max(rmse_step.fext2, [], 'omitnan') ...
    );

    eye_metrics = struct( ...
        'avg_rmse_rt', mean(rmse_eye.rt, 'omitnan'), 'min_rmse_rt', min(rmse_eye.rt, [], 'omitnan'), 'max_rmse_rt', max(rmse_eye.rt, [], 'omitnan'), ...
        'avg_rmse_ft', mean(rmse_eye.ft, 'omitnan'), 'min_rmse_ft', min(rmse_eye.ft, [], 'omitnan'), 'max_rmse_ft', max(rmse_eye.ft, [], 'omitnan'), ...
        'avg_rmse_eye_height', mean(rmse_eye.height, 'omitnan'), 'min_rmse_eye_height', min(rmse_eye.height, [], 'omitnan'), 'max_rmse_eye_height', max(rmse_eye.height, [], 'omitnan'), ...
        'avg_rmse_eye_width', mean(rmse_eye.width, 'omitnan'), 'min_rmse_eye_width', min(rmse_eye.width, [], 'omitnan'), 'max_rmse_eye_width', max(rmse_eye.width, [], 'omitnan'), ...
        'avg_rmse_eye_jitter', mean(rmse_eye.jitter, 'omitnan'), 'min_rmse_eye_jitter', min(rmse_eye.jitter, [], 'omitnan'), 'max_rmse_eye_jitter', max(rmse_eye.jitter, [], 'omitnan'), ...
        'avg_rmse_eye_amp', mean(rmse_eye.amp, 'omitnan'), 'min_rmse_eye_amp', min(rmse_eye.amp, [], 'omitnan'), 'max_rmse_eye_amp', max(rmse_eye.amp, [], 'omitnan'), ...
        'avg_mape_eye_height', mean(mape_eye.height, 'omitnan'), 'min_mape_eye_height', min(mape_eye.height, [], 'omitnan'), 'max_mape_eye_height', max(mape_eye.height, [], 'omitnan'), ...
        'avg_mape_eye_width', mean(mape_eye.width, 'omitnan'), 'min_mape_eye_width', min(mape_eye.width, [], 'omitnan'), 'max_mape_eye_width', max(mape_eye.width, [], 'omitnan') ...
    );

    fprintf("[transient evaluation] Average RMSE Step Stimulus: Main=%.4f V, NEXT1=%.4f V, FEXT1=%.4f V, NEXT2=%.4f V, FEXT2=%.4f V\n", ... 
        step_metrics.avg_rmse_main, step_metrics.avg_rmse_next1, step_metrics.avg_rmse_fext1, step_metrics.avg_rmse_next2, step_metrics.avg_rmse_fext2);
    fprintf("[transient evaluation] Min RMSE Step Stimulus: Main=%.4f V, NEXT1=%.4f V, FEXT1=%.4f V, NEXT2=%.4f V, FEXT2=%.4f V\n", ...
        step_metrics.min_rmse_main, step_metrics.min_rmse_next1, step_metrics.min_rmse_fext1, step_metrics.min_rmse_next2, step_metrics.min_rmse_fext2);
    fprintf("[transient evaluation] Max RMSE Step Stimulus: Main=%.4f V, NEXT1=%.4f V, FEXT1=%.4f V, NEXT2=%.4f V, FEXT2=%.4f V\n", ...
        step_metrics.max_rmse_main, step_metrics.max_rmse_next1, step_metrics.max_rmse_fext1, step_metrics.max_rmse_next2, step_metrics.max_rmse_fext2);

    fprintf("[transient evaluation] Average RMSE Eye Metrics: Rise Time=%.4f s, Fall Time=%.4f s, Eye Height=%.4f V, Eye Width=%.4f s, Eye Jitter=%.4f s, Eye Amplitude=%.4f V\n", ...
        eye_metrics.avg_rmse_rt, eye_metrics.avg_rmse_ft, eye_metrics.avg_rmse_eye_height, eye_metrics.avg_rmse_eye_width, eye_metrics.avg_rmse_eye_jitter, eye_metrics.avg_rmse_eye_amp);
    fprintf("[transient evaluation] Min RMSE Eye Metrics: Rise Time=%.4f s, Fall Time=%.4f s, Eye Height=%.4f V, Eye Width=%.4f s, Eye Jitter=%.4f s, Eye Amplitude=%.4f V\n", ...
        eye_metrics.min_rmse_rt, eye_metrics.min_rmse_ft, eye_metrics.min_rmse_eye_height, eye_metrics.min_rmse_eye_width, eye_metrics.min_rmse_eye_jitter, eye_metrics.min_rmse_eye_amp);
    fprintf("[transient evaluation] Max RMSE Eye Metrics: Rise Time=%.4f s, Fall Time=%.4f s, Eye Height=%.4f V, Eye Width=%.4f s, Eye Jitter=%.4f s, Eye Amplitude=%.4f V\n", ...
        eye_metrics.max_rmse_rt, eye_metrics.max_rmse_ft, eye_metrics.max_rmse_eye_height, eye_metrics.max_rmse_eye_width, eye_metrics.max_rmse_eye_jitter, eye_metrics.max_rmse_eye_amp);
    fprintf("[transient evaluation] Average MAPE: Eye Height=%.2f%%, Eye Width=%.2f%%\n", eye_metrics.avg_mape_eye_height, eye_metrics.avg_mape_eye_width);
end