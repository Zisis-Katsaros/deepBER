function h_t = process_channel(V_freq, half_win)
    V_freq = V_freq(:);
    V_win = V_freq .* half_win;
    V_sym = [V_win; conj(flip(V_win(2:end)))];
    h_raw = ifft(V_sym, 'symmetric');
    
    half_idx = floor(length(h_raw) / 2);
    h_raw(half_idx:end) = 0; % Enforce causality natively
    h_t = h_raw;
end