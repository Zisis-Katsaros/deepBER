function [eye_hight, eye_jitter, eye_amp] = calculate_eye_stats(eye_matrix, fs, bit_rate, search_window_perc)
    arguments
        eye_matrix
        fs
        bit_rate
        search_window_perc (1,1) double = 0.2 % Default search window is 20% of the bit period
    end

    bit_period = 1/bit_rate;
    samples_per_bit = round(bit_period * fs);

    center_idx = samples_per_bit; % Center of the eye diagram (1 UI)
    v_center = eye_matrix(center_idx, :); % Voltage at the center of the eye diagram

    v_thresh = mean(v_center); % Threshold voltage (mean of center voltages)
    v_high = v_center(v_center > v_thresh); % High voltages at the center
    v_low = v_center(v_center < v_thresh); % Low voltages at the center

    eye_amp = mean(v_high) - mean(v_low); % Eye amplitude (difference between mean high and low voltages)
    eye_hight = min(v_high) - max(v_low); % Eye height (minimum high voltage minus maximum low voltage)
    if eye_hight < 0
        eye_hight = 0; % Eye if fully closed
    end

    cross_idx = round(samples_per_bit / 2); % Index for crossing point (0.5 UI)
    search_window = floor(samples_per_bit * search_window_perc); % Search window based on percentage

    crossing_times = [];
    for col = 1:size(eye_matrix, 2)
        trace = eye_matrix(cross_idx - search_window : cross_idx + search_window, col);
        crossing_idx = find(diff(sign(trace - v_thresh)), 1);
        if ~isempty(crossing_idx)
            crossing_times(end+1) = crossing_idx / fs; % Convert index to time
        end
    end

    if ~isempty(crossing_times)
        eye_jitter = max(crossing_times) - min(crossing_times); % Eye jitter (range of crossing times)
    else
        eye_jitter = NaN; % No crossings found
    end
