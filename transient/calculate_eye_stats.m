function [eye_height, eye_jitter, eye_amp, eye_width] = calculate_eye_stats(eye_matrix, fs, bit_rate, search_window_perc)
    arguments
        eye_matrix
        fs
        bit_rate
        search_window_perc (1,1) double = 0.25 
    end

    bit_period = 1/bit_rate;
    samples_per_bit = round(bit_period * fs);

    center_idx = samples_per_bit; 
    v_center = eye_matrix(center_idx, :); 

    v_thresh = mean(v_center); 
    v_high = v_center(v_center > v_thresh); 
    v_low = v_center(v_center < v_thresh); 

    % Fallback if the signal has zero variance (completely flat)
    if isempty(v_high) || isempty(v_low)
        eye_amp = 0; 
        eye_height = 0; 
        eye_jitter = bit_period; 
        eye_width = 0;
        return;
    end

    eye_amp = mean(v_high) - mean(v_low); 
    
    % Statistical closure check
    stat_inner_high = mean(v_high) - 2 * std(v_high);
    stat_inner_low = mean(v_low) + 2 * std(v_low);
    
    if stat_inner_high <= stat_inner_low
        eye_height = 0;
    else
        eye_height = min(v_high) - max(v_low); 
        if eye_height < (0.02 * eye_amp) 
            eye_height = 0; 
        end
    end

    cross_idx = round(samples_per_bit / 2); 
    search_window = floor(samples_per_bit * search_window_perc); 

    crossing_times = [];
    for col = 1:size(eye_matrix, 2)
        start_idx = max(1, cross_idx - search_window);
        end_idx = min(size(eye_matrix, 1), cross_idx + search_window);
        trace = eye_matrix(start_idx : end_idx, col);
        
        crossing_rel_idx = find(diff(sign(trace - v_thresh)), 1);
        if ~isempty(crossing_rel_idx)
            crossing_abs_idx = start_idx + crossing_rel_idx - 1;
            crossing_times(end+1) = crossing_abs_idx / fs; 
        end
    end

    % If crossings exist, calculate jitter. If height is 0, width is forced to 0.
    if length(crossing_times) > 1
        eye_jitter = max(crossing_times) - min(crossing_times); 
        
        if eye_height == 0
            eye_width = 0;
        else
            eye_width = bit_period - eye_jitter; 
            if eye_width < 0
                eye_width = 0;
            end
        end
    else
        % No crossings found means the eye is completely collapsed horizontally
        eye_jitter = bit_period; 
        eye_width = 0; 
    end
end