function plot_eye_pred_vs_act(t, eye_matrix_Vout_pred, eye_matrix_Vout_actual, plot_title, eye_matrix_Vout_pred_adj)
    arguments
        t
        eye_matrix_Vout_pred
        eye_matrix_Vout_actual
        plot_title (1,1) string = "Eye Diagram Prediction Vs Actual"
        eye_matrix_Vout_pred_adj = []
    end
    
    figure('Name', 'Receiver Eye Diagram');
    
    % Plot Actual First
    plot(t, eye_matrix_Vout_actual, 'r', 'Color', [1 0 0 0.1]); 
    hold on;
    grid on;
    
    if ~isempty(eye_matrix_Vout_pred_adj)
        plot(t, eye_matrix_Vout_pred, 'b', 'Color', [0 0 1 0.05]); % Lighter opacity for raw
        plot(t, eye_matrix_Vout_pred_adj, 'g', 'Color', [0 1 0 0.1]); % Green for adjusted
        
        h_leg_actual   = plot(NaN, NaN, 'r', 'LineWidth', 2); 
        h_leg_pred_raw = plot(NaN, NaN, 'b', 'LineWidth', 2); 
        h_leg_pred_adj = plot(NaN, NaN, 'g', 'LineWidth', 2); 
        legend([h_leg_actual, h_leg_pred_raw, h_leg_pred_adj], {'Actual', 'Predicted (Raw)', 'Predicted (Adjusted)'}, 'Location', 'best');
    else
        plot(t, eye_matrix_Vout_pred, 'b', 'Color', [0 0 1 0.1]); 
        
        h_leg_actual = plot(NaN, NaN, 'r', 'LineWidth', 2); 
        h_leg_pred   = plot(NaN, NaN, 'b', 'LineWidth', 2); 
        legend([h_leg_actual, h_leg_pred], {'Actual', 'Predicted'}, 'Location', 'best');
    end
    
    title(sprintf('%s', plot_title));
    xlabel('Time (UI)');
    ylabel('Voltage (V)');
    xlim([0 2]);
    hold off;
end