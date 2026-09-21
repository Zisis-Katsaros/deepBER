function [aligned_start, aligned_end] = get_eye_alignment(num_bits, samples_per_bit, V_out, settle_bits)
    arguments
        num_bits (1,1) {mustBeInteger, mustBePositive}
        samples_per_bit (1,1) {mustBeInteger, mustBePositive}
        V_out (1,:) {mustBeNumeric}
        settle_bits (1,1) {mustBeInteger, mustBeNonnegative} = 10
    end
     % Force valid_bits to be an even number to allow reshaping into 2-UI columns
    valid_bits = floor((num_bits - settle_bits - 1) / 2) * 2; 
    
    valid_samples = valid_bits * samples_per_bit; 
    settle_idx = settle_bits * samples_per_bit;

    % Check variance across columns temporarily to find the true first crossing point
    temp_matrix = reshape(V_out(settle_idx + 1 : settle_idx + valid_samples), samples_per_bit * 2, []);
    v_var = var(temp_matrix, 0, 2);
    [~, actual_cross_idx] = min(v_var(1:samples_per_bit));
    
    % Calculate shift required to perfectly align crossing to exactly 0.5 UI
    target_cross_idx = round(samples_per_bit / 2);
    idx_offset = actual_cross_idx - target_cross_idx;

    % Shift the linear read-window. This naturally absorbs the delay and centers the eye geometry natively!
    aligned_start = settle_idx + 1 + idx_offset;
    aligned_end = aligned_start + valid_samples - 1;
end