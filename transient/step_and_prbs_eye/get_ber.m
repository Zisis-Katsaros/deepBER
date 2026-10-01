function [stat_ber, best_Q] = get_ber(V_in, V_out, bit_rate, num_bits, error_free_threshold)
    arguments
        V_in (:,1) double
        V_out (:,1) double
        bit_rate (1,1) double {mustBePositive}
        num_bits (1,1) double {mustBePositive}
        error_free_threshold (1,1) double {mustBeNonnegative} = 0
    end
    bit_period = 1 / bit_rate;
    num_samples = length(V_out);

    % Generate Time Vector
    t_prbs = linspace(0, num_bits * bit_period, num_samples)';
    Ts = t_prbs(2) - t_prbs(1);

    % Determine the Time of Flight (delay) via cross-correlation
    [c, lags] = xcorr(V_out - mean(V_out), V_in - mean(V_in));
    [~, max_idx] = max(c);
    base_delay = lags(max_idx) * Ts;

    % Dynamic Tx Threshold (handles different UCIe VDD swings)
    vref_in = (max(V_in) + min(V_in)) / 2;
    
    % Sweep phase around the xcorr delay to find the optimal sampling point
    % (Simulates Rx phase centering)
    phase_offsets = linspace(-0.2*bit_period, 0.2*bit_period, 21);
    best_Q = 0;
    
    for i = 1:length(phase_offsets)
        current_delay = base_delay + phase_offsets(i);
        t_centers = (0.5 : 1 : num_bits-0.5)' * bit_period;
        
        % Extract Tx bits
        tx_bits = interp1(t_prbs, V_in, t_centers) > vref_in;
        
        % Extract Rx voltages
        rx_samples = interp1(t_prbs, V_out, t_centers + current_delay);
        
        valid_idx = ~isnan(rx_samples);
        temp_rx = rx_samples(valid_idx);
        temp_tx = tx_bits(valid_idx);
        
        v1 = temp_rx(temp_tx == 1);
        v0 = temp_rx(temp_tx == 0);
        
        % Check if we have enough samples to calculate statistics
        if isempty(v1) || isempty(v0)
            continue;
        end
        
        mu1 = mean(v1); sigma1 = std(v1);
        mu0 = mean(v0); sigma0 = std(v0);
        
        % Calculate Q-factor
        Q_factor = (mu1 - mu0) / (sigma1 + sigma0 + 1e-15);
        
        % Keep the best Q-factor (widest vertical eye opening)
        if Q_factor > best_Q
            best_Q = Q_factor;
        end
    end

    % If the signal is completely degraded, best_Q might be 0 or negative
    if best_Q <= 0
        stat_ber = 0.5; % Maximum error rate
    else
        % Calculate optimal statistical BER
        stat_ber = 0.5 * erfc(best_Q / sqrt(2));

        if stat_ber < error_free_threshold
            stat_ber = error_free_threshold; % Set to threshold if below, considered error-free
        end
    end
end