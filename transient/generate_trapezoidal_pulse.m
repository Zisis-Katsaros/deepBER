function V_in = generate_trapezoidal_pulse(t, Vhi, bit_period, rise_time)
    % Generates a single 0 -> Vhi -> 0 trapezoidal pulse.
    V_in = zeros(size(t));
    for i = 1:length(t)
        if t(i) <= rise_time
            V_in(i) = Vhi * (t(i) / rise_time);
        elseif t(i) <= bit_period
            V_in(i) = Vhi;
        elseif t(i) <= bit_period + rise_time
            V_in(i) = Vhi * (1 - (t(i) - bit_period) / rise_time);
        end
    end
end