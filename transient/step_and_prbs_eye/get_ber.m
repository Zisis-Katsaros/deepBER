function stat_ber = get_ber(V_in, V_out, bit_rate, num_bits)
    bit_period = 1 / bit_rate;
    num_samples = length(V_out);

    % Generate Time Vector
    t_prbs = linspace(0, num_bits * bit_period, num_samples)';
    Ts = t_prbs(2) - t_prbs(1);

    % Determine the Time of Flight (delay) via cross-correlation
    [c, lags] = xcorr(V_out - mean(V_out), V_in - mean(V_in));
    [~, max_idx] = max(c);
    delay_time = lags(max_idx) * Ts;

    % Calculate exact bit center times based on UI
    t_centers = (0.5 : 1 : num_bits-0.5)' * bit_period;

    % Extract transmitted bits from the input stimulus at bit centers
    tx_bits = interp1(t_prbs, V_in, t_centers) > 0.5;

    % Extract received voltages shifted by the channel delay
    rx_samples = interp1(t_prbs, V_out, t_centers + delay_time);

    % Filter out NaN values (bits that fall outside the simulation window due to delay)
    valid_idx = ~isnan(rx_samples);
    rx_samples = rx_samples(valid_idx);
    tx_bits = tx_bits(valid_idx);

    % Group sampled voltages by transmitted bit
    v1 = rx_samples(tx_bits == 1);
    v0 = rx_samples(tx_bits == 0);

    % Calculate Statistics
    mu1 = mean(v1);
    sigma1 = std(v1);
    mu0 = mean(v0);
    sigma0 = std(v0);

    % Statistical BER based on Q-factor
    % Q = (mu1 - mu0) / (sigma1 + sigma0)
    Q_factor = (mu1 - mu0) / (sigma1 + sigma0 + 1e-15);
    stat_ber = 0.5 * erfc(Q_factor / sqrt(2));
end





