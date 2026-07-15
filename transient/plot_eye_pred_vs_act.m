function plot_eye_pred_vs_act(t, eye_matrix_Vout_pred, eye_matrix_Vout_actual, plot_title)
    arguments
        t
        eye_matrix_Vout_pred
        eye_matrix_Vout_actual
        plot_title (1,1) string = "Eye Diagram Prediction Vs Actual"
    end
    
    figure('Name', 'Receiver Eye Diagram');
    plot(t, eye_matrix_Vout_pred, 'b', 'Color', [0 0 1 0.1]); % 10% opacity for density effect
    hold on;
    plot(t, eye_matrix_Vout_actual, 'r', 'Color', [1 0 0 0.1]); % 10% opacity for density effect
    grid on;
    title(sprintf('%s', plot_title));
    xlabel('Time (UI)');
    ylabel('Voltage (V)');

    % Dummy plots for legend, since low opacity makes colours hard to distinguish in legend
    h_leg_actual = plot(NaN, NaN, 'r', 'LineWidth', 2); % Solid Red
    h_leg_pred   = plot(NaN, NaN, 'b', 'LineWidth', 2); % Solid Blue
    legend([h_leg_pred, h_leg_actual], {'Predicted', 'Actual'}, 'Location', 'best');
    xlim([0 2]);
end