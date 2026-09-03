function V_out_total = apply_xtalk(fit_main, xtalk_fits, V_in_prbs, Vhi, Ts, samples_per_bit, type, alpha_correction)
    arguments
        fit_main (1,1)
        xtalk_fits (1,:) cell
        V_in_prbs (:,1) double
        Vhi (1,1) double
        Ts (1,1) double
        samples_per_bit (1,1) double
        type (1,1) string {mustBeMember(type, ["worst-case", "realistic"])} = "realistic"
        alpha_correction (1,1) double = 1.0
    end
    
    % Main channel response
    V_out_main = timeresp(fit_main, V_in_prbs, Ts) * alpha_correction;
    V_out_total = V_out_main;
    
    % Sum the crosstalk contributions from all active adjacent lines
    for k = 1:length(xtalk_fits)
        if ~isempty(xtalk_fits{k})
            if type == "worst-case"
                % Invert the stimulus for aggressors.
                % This ensures that when the main channel transitions 0 -> Vhi,
                % the aggressors transition Vhi -> 0, generating maximum opposing crosstalk.
                V_in_aggressor = Vhi - V_in_prbs;
            else
                % Shift by a large prime number of UIs (e.g., 89) for each aggressor to ensure statistical independence
                shift_amount = 89 * k * samples_per_bit;
                V_in_aggressor = circshift(V_in_prbs, shift_amount);
            end
            V_out_xtalk = timeresp(xtalk_fits{k}, V_in_aggressor, Ts) * alpha_correction;
            V_out_total = V_out_total + V_out_xtalk;
        end
    end
end