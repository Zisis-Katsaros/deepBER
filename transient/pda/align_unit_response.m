function [y_aligned, shift_samples] = align_unit_response(y, samples_per_bit)
    % Find the absolute peak to separate the rising and falling edges
    [~, peak_idx] = max(y);
    
    % Define half amplitude to find the crossings
    half_amp = max(y) / 2;
    
    % Find the Rising Edge crossing (search backwards from peak)
    search_rise = max(1, peak_idx - samples_per_bit) : peak_idx;
    [~, rise_offset] = min(abs(y(search_rise) - half_amp));
    rise_idx = search_rise(rise_offset);
    
    % Find the Falling Edge crossing (search forwards from peak, allowing 2 UI for long tails)
    search_fall = peak_idx : min(length(y), peak_idx + 2 * samples_per_bit);
    [~, fall_offset] = min(abs(y(search_fall) - half_amp));
    fall_idx = search_fall(fall_offset);
    
    % Calculate the true center of the eye (midpoint between crossings)
    eye_center_idx = round((rise_idx + fall_idx) / 2);
    
    % Target the exact middle of the 1-UI column
    target_idx = round(samples_per_bit / 2);
    
    % Find where the eye center currently lands and calculate the shift
    current_idx_in_ui = mod(eye_center_idx - 1, samples_per_bit) + 1;
    shift_samples = target_idx - current_idx_in_ui;
    
    % Circularly shift y_main so the entire eye is perfectly encapsulated
    y_aligned = circshift(y, shift_samples);