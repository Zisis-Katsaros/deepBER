function [eye_height, eye_jitter, eye_amp, eye_width] = calculate_eye_stats(eye_matrix, fs, bit_rate, search_window_perc, check_for_closure)
    arguments
        eye_matrix
        fs
        bit_rate
        search_window_perc (1,1) double = 0.25 
        check_for_closure (1,1) logical = false
    end

    bit_period = 1/bit_rate;
    samples_per_bit = round(bit_period * fs);

    %{
    center_idx = samples_per_bit; 
    v_center = eye_matrix(center_idx, :); 

    v_thresh = mean(v_center); 
    v_high = v_center(v_center > v_thresh); 
    v_low = v_center(v_center < v_thresh); 
    %}

    nom_center_idx = samples_per_bit; 
    
    % Define a search window to find the optimal sampling phase (the widest vertical opening)
    search_radius = round(0.30 * samples_per_bit);
    search_range = (nom_center_idx - search_radius) : (nom_center_idx + search_radius);
    
    v_thresh = mean(mean(eye_matrix)); % Global threshold
    
    max_eye_height = 0;
    optimal_idx = nom_center_idx;
    best_v_high = [];
    best_v_low = [];
    
    % Sweep across the center of the eye to find the maximum inner contour (like PDA)
    for idx = search_range
        v_slice = eye_matrix(idx, :);
        
        v_high = v_slice(v_slice > v_thresh);
        v_low  = v_slice(v_slice < v_thresh);
        
        if ~isempty(v_high) && ~isempty(v_low)
            % Inner contour distance at this specific time slice
            current_height = min(v_high) - max(v_low); 
            
            if current_height > max_eye_height
                max_eye_height = current_height;
                optimal_idx = idx;
                best_v_high = v_high;
                best_v_low = v_low;
            end
        end
    end

    % Fallback if the signal has zero variance (completely flat)
    if isempty(v_high) || isempty(v_low)
        eye_amp = 0; 
        eye_height = 0; 
        eye_jitter = bit_period; 
        eye_width = 0;
        return;
    end

    % Proceed with the best values found at the optimal sampling phase
    eye_amp = mean(best_v_high) - mean(best_v_low); 
    eye_height = max_eye_height;
    
    % Statistical closure check (using the optimal slice)
    stat_inner_high = mean(best_v_high) - 2 * std(best_v_high);
    stat_inner_low = mean(best_v_low) + 2 * std(best_v_low);
    
    if check_for_closure && stat_inner_high <= stat_inner_low
        eye_height = 0;
    end

    %{
    eye_amp = mean(v_high) - mean(v_low); 
    
    % Statistical closure check
    stat_inner_high = mean(v_high) - 2 * std(v_high);
    stat_inner_low = mean(v_low) + 2 * std(v_low);
    
    if check_for_closure && stat_inner_high <= stat_inner_low
        eye_height = 0;
    else
        eye_height = min(v_high) - max(v_low); 
        if eye_height < (0.02 * eye_amp) 
            eye_height = 0; 
        end
    end
    %}
    
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