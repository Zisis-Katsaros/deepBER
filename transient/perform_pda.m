function [s1, s0, metrics] = perform_pda(fit_main, xtalk_fits, Ts, bit_rate, Vhi, mask_height, mask_width, alpha_correction)
    % Perform Peak Distortion Analysis (PDA) on a given main channel and its crosstalk channels.
    
    samples_per_bit = round((1/bit_rate) / Ts);
    num_bits = 200; % Simulate 200 UI to allow all ringing/reflections to settle
    
    % Create Unit Pulse
    t_sim = (0:Ts:(num_bits*samples_per_bit-1)*Ts)';
    rise_time = 15e-12; % 15 ps realistic rise time
    V_pulse = generate_trapezoidal_pulse(t_sim, Vhi, 1/bit_rate, rise_time);
    
    % Get Unit Pulse Responses
    y_main = timeresp(fit_main, V_pulse, Ts) * alpha_correction;
    
    % Fold the response into a matrix of 1 UI columns
    y_main_fold = reshape(y_main, samples_per_bit, num_bits);
    
    % Isolate the Main Cursor (the bit interval with the max peak)
    [~, cursor_idx] = max(max(y_main_fold));
    y_cursor = y_main_fold(:, cursor_idx);
    
    % Calculate Intersymbol Interference (ISI)
    y_isi = y_main_fold;
    y_isi(:, cursor_idx) = 0; % Remove the main cursor from the ISI matrix
    
    isi_neg = sum(min(0, y_isi), 2); % Sum of all negative ISI ringing
    isi_pos = sum(max(0, y_isi), 2); % Sum of all positive ISI ringing
    
    % Calculate Cochannel Interference (Crosstalk)
    xtalk_neg = zeros(samples_per_bit, 1);
    xtalk_pos = zeros(samples_per_bit, 1);
    
    for k = 1:length(xtalk_fits)
        if ~isempty(xtalk_fits{k})
            y_xtalk = timeresp(xtalk_fits{k}, V_pulse, Ts) * alpha_correction;
            y_xtalk_fold = reshape(y_xtalk, samples_per_bit, num_bits);
            
            % Summing min(0, xtalk) naturally bounds the scenario where the adjacent 
            % line switches in the direction needed to ruin the main signal.
            xtalk_neg = xtalk_neg + sum(min(0, y_xtalk_fold), 2);
            xtalk_pos = xtalk_pos + sum(max(0, y_xtalk_fold), 2);
        end
    end
    
    % Construct Worst-Case Eye Edges
    s1 = y_cursor + isi_neg + xtalk_neg;
    s0 = isi_pos + xtalk_pos;
    
    % Extract Metrics
    metrics = struct();
    metrics.eye_height = max(0, max(s1 - s0)); % Ensure it doesn't go below 0
    metrics.eye_amp = max(y_cursor);
    
    % Calculate Eye Width and Jitter at V_ref
    if metrics.eye_height > 0
        Vref = (max(s1) + min(s0)) / 2;
        valid_width_indices = (s1 > Vref) & (s0 < Vref);
        metrics.eye_width = sum(valid_width_indices) * Ts;
        metrics.jitter = (1/bit_rate) - metrics.eye_width;
    else
        metrics.eye_width = 0;
        metrics.jitter = 1/bit_rate;
    end
    
    % Check the UCIe Target Mask Rectangle
    win_samples = round(mask_width / Ts);
    metrics.passes_mask = false;
    
    max_margin = -inf;
    best_win_start = 1;
    best_min_s1 = 0;
    best_max_s0 = 0;
    
    % Sliding window to see if the mask rectangle fits anywhere inside the eye
    for i = 1:(samples_per_bit - win_samples)
        local_min_s1 = min(s1(i:i+win_samples));
        local_max_s0 = max(s0(i:i+win_samples));
        margin = local_min_s1 - local_max_s0;
        
        if margin > max_margin
            max_margin = margin;
            best_win_start = i;
            best_min_s1 = local_min_s1;
            best_max_s0 = local_max_s0;
        end
    end
    
    if max_margin >= mask_height
        metrics.passes_mask = true;
    end
    
    % Save the optimal mask center coordinates for plotting
    metrics.mask_center_idx = best_win_start + round(win_samples / 2);
    metrics.mask_center_v = (best_min_s1 + best_max_s0) / 2;
end