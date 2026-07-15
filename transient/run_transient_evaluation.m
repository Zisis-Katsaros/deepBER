function run_transient_evaluation(filename_preds, filename_actuals,  title, fs, t_step, rise_time, delay, Vhi, num_bits, bit_rate, precision)
    arguments
        filename_preds (1,1) string
        filename_actuals (1,1) string
        title (1,1) string = "Transient Evaluation"
        fs (1,1) double = 1e12
        t_step (1,1) double = 2e-9
        rise_time (1,1) double {mustBePositive} = 40e-12; % Default rise time is 40 ps
        delay (1,1) double {mustBeNonnegative} = 100e-12; % Default delay is 100 ps
        Vhi (1,1) double {mustBePositive} = 1; % Default high voltage is 1V
        num_bits (1,1) double {mustBeInteger, mustBePositive} = 1000; % Default number of bits is 1000
        bit_rate (1,1) double {mustBePositive} = 10e9; % Default bit rate is 10 Gbps
        precision = -40; % Default precision is -40 dB  
    end

    % Setup time vectors 
    Ts = 1/fs; % sampling period
    t = (0:Ts:t_step)';

    % Create 0->1 step stimulus
    V_in_step = lo2hi_step_stimulus(t, rise_time, delay, Vhi);

    % Create PRBS strimulus
    V_in_prbs = prbs_stimulus(num_bits, bit_rate, rise_time, Ts);
    bit_period = 1/bit_rate;

    % Initialze RMSE accumulators
    total_rmse_main = 0;
    total_rmse_next1 = 0;
    total_rmse_fext1 = 0;
    total_rmse_next2 = 0;
    total_rmse_fext2 = 0;

    min_rmse_main = inf;
    min_rmse_next1 = inf;
    min_rmse_fext1 = inf;
    min_rmse_next2 = inf;
    min_rmse_fext2 = inf;

    max_rmse_main = -inf;
    max_rmse_next1 = -inf;
    max_rmse_fext1 = -inf;
    max_rmse_next2 = -inf;
    max_rmse_fext2 = -inf;

    % Initialize Eye metrics accumulators
    total_rmse_rt = 0;
    total_rmse_ft = 0;
    total_rmse_eye_hight = 0;
    total_rmse_eye_jitter = 0;
    total_rmse_eye_amp = 0;

    min_rmse_rt = inf;
    min_rmse_ft = inf;
    min_rmse_eye_hight = inf;
    min_rmse_eye_jitter = inf;
    min_rmse_eye_amp = inf;

    max_rmse_rt = -inf;
    max_rmse_ft = -inf;
    max_rmse_eye_hight = -inf;
    max_rmse_eye_jitter = -inf;
    max_rmse_eye_amp = -inf;

    fprintf("[transient evaluation] Beginning transient evaluation for geometry: \n");
    for port = 1:9
        fprintf("[transient evaluation] \tEvaluating port %d...\n", port);
        % Main channel ports
        tx = port;
        rx = tx + 9;

        % Next channel ports
        next1 = tx - 1;
        next2 = tx + 1;
        % Far-end channel ports
        fext1 = rx - 1;
        fext2 = rx + 1;

        % Placeholders for ports 1 & 9, since there is only one adjacent port for outer ports
        if port == 1
            next1 = tx;
            fext1 = rx;      
        elseif port == 9
            next2 = tx;
            fext2 = rx;
        end
        % Convert S-parameters to time-domain impulse responses
        [fit_main_pred, fit_next1_pred, fit_fext1_pred, fit_next2_pred, fit_fext2_pred] = s_params2impulse_response(filename_preds, tx, rx, ...
        next1, fext1, next2, fext2, precision);
        [fit_main_actual, fit_next1_actual, fit_fext1_actual, fit_next2_actual, fit_fext2_actual] = s_params2impulse_response(filename_actuals, tx, rx, ...
        next1, fext1, next2, fext2, precision);

        % Evaludate step responses
        % Prediction
        V_out_main_step_pred = timeresp(fit_main_pred, V_in_step, Ts) / 2;
        V_out_next1_step_pred = timeresp(fit_next1_pred, V_in_step, Ts) / 2;
        V_out_fext1_step_pred = timeresp(fit_fext1_pred, V_in_step, Ts) / 2;
        V_out_next2_step_pred = timeresp(fit_next2_pred, V_in_step, Ts) / 2;
        V_out_fext2_step_pred = timeresp(fit_fext2_pred, V_in_step, Ts) / 2;

        % Actual
        V_out_main_step_actual = timeresp(fit_main_actual, V_in_step, Ts) / 2;
        V_out_next1_step_actual = timeresp(fit_next1_actual, V_in_step, Ts) / 2;
        V_out_fext1_step_actual = timeresp(fit_fext1_actual, V_in_step, Ts) / 2;
        V_out_next2_step_actual = timeresp(fit_next2_actual, V_in_step, Ts) / 2;
        V_out_fext2_step_actual = timeresp(fit_fext2_actual, V_in_step, Ts) / 2;

        % Overwrite for ports 1 and 9
        if port == 1
            V_out_next1_step_pred = [];
            V_out_fext1_step_pred = [];
            V_out_next1_step_actual = [];
            V_out_fext1_step_actual = [];
        elseif port == 9
            V_out_next2_step_pred = [];
            V_out_fext2_step_pred = [];
            V_out_next2_step_actual = [];
            V_out_fext2_step_actual = [];
        end

        % Step response RMSE
        rmse_main = rmse(V_out_main_step_pred, V_out_main_step_actual);
        rmse_next1 = rmse(V_out_next1_step_pred, V_out_next1_step_actual);
        rmse_fext1 = rmse(V_out_fext1_step_pred, V_out_fext1_step_actual);
        rmse_next2 = rmse(V_out_next2_step_pred, V_out_next2_step_actual);
        rmse_fext2 = rmse(V_out_fext2_step_pred, V_out_fext2_step_actual);

        total_rmse_main = total_rmse_main + rmse_main;
        if rmse_main < min_rmse_main
            min_rmse_main = rmse_main;
        end
        if rmse_main > max_rmse_main
            max_rmse_main = rmse_main;
        end

        fprintf("[transient evaluation] \t>> 0->1 Step stimulus:\n");
        fprintf("[transient evaluation] \t- RMSE (Main): %.4f V\n", rmse_main);
        if port ~= 1
            fprintf("[transient evaluation] \t- RMSE (NEXT1): %.4f V\n", rmse_next1);
            fprintf("[transient evaluation] \t- RMSE (FEXT1): %.4f V\n", rmse_fext1);

            total_rmse_next1 = total_rmse_next1 + rmse_next1;
            total_rmse_fext1 = total_rmse_fext1 + rmse_fext1;
            if rmse_next1 < min_rmse_next1
                min_rmse_next1 = rmse_next1;
            end
            if rmse_next1 > max_rmse_next1
                max_rmse_next1 = rmse_next1;
            end

            if rmse_fext1 < min_rmse_fext1
                min_rmse_fext1 = rmse_fext1;
            end
            if rmse_fext1 > max_rmse_fext1
                max_rmse_fext1 = rmse_fext1;
            end
        end 
        if port ~= 9
            fprintf("[transient evaluation] \t- RMSE (NEXT2): %.4f V\n", rmse_next2);
            fprintf("[transient evaluation] \t- RMSE (FEXT2): %.4f V\n", rmse_fext2);

            total_rmse_next2 = total_rmse_next2 + rmse_next2;
            total_rmse_fext2 = total_rmse_fext2 + rmse_fext2;
            if rmse_next2 < min_rmse_next2
                min_rmse_next2 = rmse_next2;
            end
            if rmse_next2 > max_rmse_next2
                max_rmse_next2 = rmse_next2;
            end
            if rmse_fext2 < min_rmse_fext2
                min_rmse_fext2 = rmse_fext2;
            end
            if rmse_fext2 > max_rmse_fext2
                max_rmse_fext2 = rmse_fext2;
            end
        end
        % Plot step responses
        plot_step_response_pred_vs_act(t, V_in_step, V_out_main_step_pred, V_out_main_step_actual, ...
            V_out_next1_step_pred, V_out_next1_step_actual, V_out_fext1_step_pred, V_out_fext1_step_actual, ...
            V_out_next2_step_pred, V_out_next2_step_actual, V_out_fext2_step_pred, V_out_fext2_step_actual, ...
            sprintf('%s - Step Response Prediction Vs Actual (Port %d)', title, port));

        % Evaluate PRBS responses
        V_out_main_prbs_pred = timeresp(fit_main_pred, V_in_prbs, Ts) / 2;
        V_out_main_prbs_actual = timeresp(fit_main_actual, V_in_prbs, Ts) / 2;

        % Reshape the PRBS vector to fold it over itself based on the bit period
        samples_per_bit = round(bit_period * fs);
        % Discard the first few bits to allow the transient to settle
        settle_bits = 10;
        valid_samples = (num_bits - settle_bits) * samples_per_bit;

        % Reshape into a matrix where each column is 2 bit periods wide (standard for eye diagrams)
        eye_matrix_Vout_pred = reshape(V_out_main_prbs_pred(settle_bits*samples_per_bit + 1 : settle_bits*samples_per_bit + valid_samples), ...
                                samples_per_bit * 2, []);
        eye_matrix_Vout_actual = reshape(V_out_main_prbs_actual(settle_bits*samples_per_bit + 1 : settle_bits*samples_per_bit + valid_samples), ...
                                samples_per_bit * 2, []);
        
        % Calculate eye metrics
        [rmse_rt, rmse_ft, rmse_eye_hight, rmse_eye_jitter, rmse_eye_amp] = eye_metrics_pred_vs_act(V_out_main_prbs_pred, V_out_main_prbs_actual, ...
            eye_matrix_Vout_pred, eye_matrix_Vout_actual, fs, bit_rate);

        total_rmse_rt = total_rmse_rt + rmse_rt;
        total_rmse_ft = total_rmse_ft + rmse_ft;
        total_rmse_eye_hight = total_rmse_eye_hight + rmse_eye_hight;
        total_rmse_eye_jitter = total_rmse_eye_jitter + rmse_eye_jitter;
        total_rmse_eye_amp = total_rmse_eye_amp + rmse_eye_amp;

        if rmse_rt < min_rmse_rt
            min_rmse_rt = rmse_rt;
        end
        if rmse_rt > max_rmse_rt
            max_rmse_rt = rmse_rt;
        end
        if rmse_ft < min_rmse_ft
            min_rmse_ft = rmse_ft;
        end
        if rmse_ft > max_rmse_ft
            max_rmse_ft = rmse_ft;
        end
        if rmse_eye_hight < min_rmse_eye_hight
            min_rmse_eye_hight = rmse_eye_hight;
        end
        if rmse_eye_hight > max_rmse_eye_hight
            max_rmse_eye_hight = rmse_eye_hight;
        end
        if rmse_eye_jitter < min_rmse_eye_jitter
            min_rmse_eye_jitter = rmse_eye_jitter;
        end
        if rmse_eye_jitter > max_rmse_eye_jitter
            max_rmse_eye_jitter = rmse_eye_jitter;
        end
        if rmse_eye_amp < min_rmse_eye_amp
            min_rmse_eye_amp = rmse_eye_amp;
        end
        if rmse_eye_amp > max_rmse_eye_amp
            max_rmse_eye_amp = rmse_eye_amp;
        end
        fprintf("[transient evaluation] \t>> PRBS stimulus:\n");
        fprintf("[transient evaluation] \t- RMSE Rise Time: %.4f s\n", rmse_rt);
        fprintf("[transient evaluation] \t- RMSE Fall Time: %.4f s\n", rmse_ft);
        fprintf("[transient evaluation] \t- RMSE Eye Height: %.4f V\n", rmse_eye_hight);
        fprintf("[transient evaluation] \t- RMSE Eye Jitter: %.4f s\n", rmse_eye_jitter);
        fprintf("[transient evaluation] \t- RMSE Eye Amplitude: %.4f V\n", rmse_eye_amp);
    
        t_eye = linspace(0, 2, samples_per_bit * 2);
        plot_eye_pred_vs_act(t_eye, eye_matrix_Vout_pred, eye_matrix_Vout_actual, sprintf('%s - Eye Diagram Prediction Vs Actual (Port %d)', title, port));    
    end

    % Calculate average RMSE for each channel across all ports
    avg_rmse_main = total_rmse_main / 9;
    avg_rmse_next1 = total_rmse_next1 / 8;
    avg_rmse_fext1 = total_rmse_fext1 / 8;
    avg_rmse_next2 = total_rmse_next2 / 8;
    avg_rmse_fext2 = total_rmse_fext2 / 8;

    % Calculate average RMSE for eye metrics across all ports
    avg_rmse_rt = total_rmse_rt / 9;
    avg_rmse_ft = total_rmse_ft / 9;
    avg_rmse_eye_hight = total_rmse_eye_hight / 9;
    avg_rmse_eye_jitter = total_rmse_eye_jitter / 9;
    avg_rmse_eye_amp = total_rmse_eye_amp / 9;

    fprintf("[transient evaluation] Average RMSE Step Stimulus: Main=%.4f V, NEXT1=%.4f V, FEXT1=%.4f V, NEXT2=%.4f V, FEXT2=%.4f V\n", ... 
    avg_rmse_main, avg_rmse_next1, avg_rmse_fext1, avg_rmse_next2, avg_rmse_fext2);
    fprintf("[transient evaluation] Min RMSE Step Stimulus: Main=%.4f V, NEXT1=%.4f V, FEXT1=%.4f V, NEXT2=%.4f V, FEXT2=%.4f V\n", ...
    min_rmse_main, min_rmse_next1, min_rmse_fext1, min_rmse_next2, min_rmse_fext2);
    fprintf("[transient evaluation] Max RMSE Step Stimulus: Main=%.4f V, NEXT1=%.4f V, FEXT1=%.4f V, NEXT2=%.4f V, FEXT2=%.4f V\n", ...
    max_rmse_main, max_rmse_next1, max_rmse_fext1, max_rmse_next2, max_rmse_fext2);

    fprintf("[transient evaluation] Average RMSE Eye Metrics: Rise Time=%.4f s, Fall Time=%.4f s, Eye Height=%.4f V, Eye Jitter=%.4f s, Eye Amplitude=%.4f V\n", ...
    avg_rmse_rt, avg_rmse_ft, avg_rmse_eye_hight, avg_rmse_eye_jitter, avg_rmse_eye_amp);
    fprintf("[transient evaluation] Min RMSE Eye Metrics: Rise Time=%.4f s, Fall Time=%.4f s, Eye Height=%.4f V, Eye Jitter=%.4f s, Eye Amplitude=%.4f V\n", ...
    min_rmse_rt, min_rmse_ft, min_rmse_eye_hight, min_rmse_eye_jitter, min_rmse_eye_amp);
    fprintf("[transient evaluation] Max RMSE Eye Metrics: Rise Time=%.4f s, Fall Time=%.4f s, Eye Height=%.4f V, Eye Jitter=%.4f s, Eye Amplitude=%.4f V\n", ...
    max_rmse_rt, max_rmse_ft, max_rmse_eye_hight, max_rmse_eye_jitter, max_rmse_eye_amp);
end
