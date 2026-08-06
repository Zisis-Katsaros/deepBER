function run_pda_evaluation(filename_preds, filename_actuals, title_str, fs, bit_rate, Vhi, alpha_correction)
    %{
    Runs Peak Distortion Analysis (PDA) on predicted and actual S-parameters.
    Identifies the worst-case channel and evaluates pass/fail against the UCIe mask.
    
    Inputs:
    - filename_preds, filename_actuals: Paths to the Touchstone files.
    - title_str: Title for plotting/logging.
    - fs: Sampling frequency in Hz (e.g., 1e12 for 1 ps resolution).
    - bit_rate: Data rate in bits per second (e.g., 32e9 for 32 GT/s).
    - Vhi: High voltage level for the transmitter (e.g., 0.625 V).
    - alpha_correction: (Optional) A scalar from the DNN to scale the predicted amplitude.
    %}
    arguments
        filename_preds (1,1) string
        filename_actuals (1,1) string
        title_str (1,1) string = "PDA Evaluation"
        fs (1,1) double = 2e12 % 0.5 ps resolution for accurate 20ps window tracking
        bit_rate (1,1) double = 32e9 % 32 GT/s UCIe Standard
        Vhi (1,1) double = 0.625
        alpha_correction (1,1) double = 1.0 % Defaults to 1 (no correction yet)
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
    results = struct('EH_pred', {}, 'EW_pred', {}, 'Pass_pred', {}, ...
                     'EH_act', {}, 'EW_act', {}, 'Pass_act', {});

    for port = 1:9
        fprintf('\n[PDA] Evaluating Port %d...\n', port);
        
        % Map ports (Main, NEXT, FEXT)
        tx = port; 
        rx = tx + 9;
        next1 = tx - 1; 
        next2 = tx + 1;
        fext1 = rx - 1; 
        fext2 = rx + 1;
        
        if port == 1
            next1 = []; fext1 = [];
        elseif port == 9
            next2 = []; fext2 = [];
        end

        % Extract VTF rational models (Using the UCIe 30-Ohm/50-Ohm & 125fF defaults)
        [fit_main_pred, fit_next1_pred, fit_fext1_pred, fit_next2_pred, fit_fext2_pred] = ...
            s_params2vtf_models(filename_preds, tx, rx, next1, fext1, next2, fext2);
            
        [fit_main_act, fit_next1_act, fit_fext1_act, fit_next2_act, fit_fext2_act] = ...
            s_params2vtf_models(filename_actuals, tx, rx, next1, fext1, next2, fext2);

        % Group crosstalk models into cell arrays for easy passing
        xtalk_fits_pred = {fit_next1_pred, fit_fext1_pred, fit_next2_pred, fit_fext2_pred};
        xtalk_fits_act = {fit_next1_act, fit_fext1_act, fit_next2_act, fit_fext2_act};

        % Run PDA
        [s1_pred, s0_pred, metrics_pred] = perform_pda(fit_main_pred, xtalk_fits_pred, Ts, bit_rate, Vhi, mask_height, mask_width, alpha_correction);
        [s1_act, s0_act, metrics_act] = perform_pda(fit_main_act, xtalk_fits_act, Ts, bit_rate, Vhi, mask_height, mask_width, 1.0); % Actuals don't get corrected

        % Store metrics
        results(port).EH_pred = metrics_pred.eye_height;
        results(port).EW_pred = metrics_pred.eye_width;
        results(port).Pass_pred = metrics_pred.passes_mask;
        results(port).EH_act = metrics_act.eye_height;
        results(port).EW_act = metrics_act.eye_width;
        results(port).Pass_act = metrics_act.passes_mask;

        % Print Port Comparison
        fprintf('\t          PREDICTED   |   ACTUAL\n');
        fprintf('\tEye Height: %.4f V  |   %.4f V \n', metrics_pred.eye_height, metrics_act.eye_height);
        fprintf('\tEye Width:  %.2f ps  |   %.2f ps \n', metrics_pred.eye_width*1e12, metrics_act.eye_width*1e12);
        fprintf('\tJitter:     %.2f ps   |   %.2f ps \n', metrics_pred.jitter*1e12, metrics_act.jitter*1e12);
        fprintf('\tUCIe Mask:  %-9s |   %-9s \n', string(metrics_pred.passes_mask), string(metrics_act.passes_mask));
        
        plot_title_str = sprintf('%s - PDA Worst-Case Eye (Port %d)', title_str, port);
        plot_pda_eye(Ts, s1_pred, s0_pred, s1_act, s0_act, metrics_pred, metrics_act, mask_height, mask_width, plot_title_str);
        
        % Track Worst Channel (Based on Actual Eye Height)
        if metrics_act.eye_height < min_global_eye_height
            min_global_eye_height = metrics_act.eye_height;
            worst_port = port;
        end
    end

    fprintf('\n======================================================\n');
    fprintf('               WORST CHANNEL ANALYSIS\n');
    fprintf('======================================================\n');
    fprintf('>> The worst-case bottleneck is PORT %d.\n', worst_port);
    fprintf('>> Actual Eye Height: %.4f V\n', results(worst_port).EH_act);
    fprintf('>> Actual Eye Width:  %.2f ps\n', results(worst_port).EW_act*1e12);
    fprintf('>> Predicted Verdict: %s | Actual Verdict: %s\n', ...
            string(results(worst_port).Pass_pred), string(results(worst_port).Pass_act));
end