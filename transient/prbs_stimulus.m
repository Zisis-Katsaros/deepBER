function V_in_prbs = prbs_stimulus(num_bits, bit_rate, rise_time, Ts)
    %{
    Generates a pseudo-random binary sequence (PRBS) stimulus with a specified rise time and sampling period.
    Inputs:
    - num_bits: Number of bits in the PRBS sequence
    - bit_rate: Bit rate of the PRBS sequence (in bits/second)
    - rise_time: Rise time of the step (in seconds)
    - Ts: Sampling period (in seconds)
    Outputs:
    - V_in_prbs: Voltage values corresponding to the time vector t_prbs
    %}
    arguments
        num_bits (1,1) double {mustBeInteger, mustBePositive} = 1000; % Default number of bits is 1000
        bit_rate (1,1) double {mustBePositive} = 10e9; % Default bit rate is 10 Gbps
        rise_time (1,1) double {mustBePositive} = 40e-12; % Default rise time is 40 ps
        Ts (1,1) double {mustBePositive} = 1e-12; % Default sampling period is 1 ps
    end
    bit_period = 1/bit_rate;

    % Generate random 1s and 0s
    bits = randi([0 1], num_bits, 1);

    % Create the PRBS time vector
    t_prbs = (0:Ts:(num_bits*bit_period))';

    V_in_prbs = zeros(size(t_prbs));
    for i = 1:num_bits
        if bits(i) == 1
            % Superimpose a shifted step up
            V_in_prbs = V_in_prbs + min(max((t_prbs - (i-1)*bit_period) / rise_time, 0), 1);
            % Superimpose a shifted step down at the end of the bit
            V_in_prbs = V_in_prbs - min(max((t_prbs - i*bit_period) / rise_time, 0), 1);
        end
    end
end