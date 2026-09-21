function [V_in_step] = lo2hi_step_stimulus(t, rise_time, delay, Vhi)
    %{
    Generates a low-to-high step stimulus with a specified rise time and delay.
    Inputs:
    - t: Time vector (1D array)
    - rise_time: Rise time of the step (in seconds)
    - delay: Delay before the step starts (in seconds)
    - Vhi: High voltage level after the step (in volts)
    Outputs:
    - V_in_step: Voltage values corresponding to the time vector t
    %}
    arguments
        t (1,:) double
        rise_time (1,1) double {mustBePositive} = 40e-12; % Default rise time is 40 ps
        delay (1,1) double {mustBeNonnegative} = 100e-12; % Default delay is 100 ps
        Vhi (1,1) double {mustBePositive} = 1; % Default high voltage is 1V
    end
    % Create a linear ramp from 0V to Vhi
    V_in_step = Vhi * min(max((t - delay) / rise_time, 0), 1);
