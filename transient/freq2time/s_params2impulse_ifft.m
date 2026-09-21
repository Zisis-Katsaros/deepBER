function [t, h_main, h_next1, h_fext1, h_next2, h_fext2] = s_params2impulse_ifft(filename, tx, rx, next1, fext1, next2, fext2, R_tx, C_tx, R_rx, C_rx)
    arguments 
        filename (1,1) string
        tx (1,1) double {mustBeInteger, mustBePositive}
        rx (1,1) double {mustBeInteger, mustBePositive}
        next1 (1,1) double {mustBeInteger, mustBePositive}
        fext1 (1,1) double {mustBeInteger, mustBePositive}
        next2 (1,1) double {mustBeInteger, mustBePositive}
        fext2 (1,1) double {mustBeInteger, mustBePositive}
        R_tx (1,1) double {mustBePositive} = 30; 
        C_tx (1,1) double {mustBeNonnegative} = 125e-15; 
        R_rx (1,1) double {mustBePositive} = 50; 
        C_rx (1,1) double {mustBeNonnegative} = 125e-15; 
    end 
    
    S_data = sparameters(filename);
    freq = S_data.Frequencies;
    num_freq = length(freq);
    
    df = freq(2) - freq(1);
    N_sym = 2 * num_freq - 1; 
    dt = 1 / (N_sym * df);
    t = (0:N_sym-1).' * dt;

    VTF_matrix = calculate_vtf_from_sparameters(S_data, R_tx, C_tx, R_rx, C_rx);
    
    full_win = hann(2 * num_freq - 1);
    half_win = full_win(num_freq:end);

    h_main  = process_channel(squeeze(VTF_matrix(rx, tx, :)), half_win);
    h_next1 = process_channel(squeeze(VTF_matrix(next1, tx, :)), half_win);
    h_fext1 = process_channel(squeeze(VTF_matrix(fext1, tx, :)), half_win);
    h_next2 = process_channel(squeeze(VTF_matrix(next2, tx, :)), half_win);
    h_fext2 = process_channel(squeeze(VTF_matrix(fext2, tx, :)), half_win);
end