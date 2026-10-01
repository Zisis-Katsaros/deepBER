function plot_tdr(t_vec, z_pred, z_act, geometry_title)
    % Plots the predicted vs actual TDR impedance profile
    
    figure('Name', ['TDR Profile - ', char(geometry_title)]);
    plot(t_vec * 1e12, z_act, 'k-', 'LineWidth', 1.5);
    hold on;
    plot(t_vec * 1e12, z_pred, 'r--', 'LineWidth', 1.5);
    
    xlabel('Time (ps)', 'FontWeight', 'bold');
    ylabel('Impedance (\Omega)', 'FontWeight', 'bold');
    title(sprintf('Time Domain Reflectometry (TDR) - %s', char(geometry_title)));
    legend('Actual (3D EM)', 'Predicted (DNN)', 'Location', 'best');
    grid on;
    
    % Optional styling for professional appearance
    set(gca, 'FontSize', 11);
    hold off;
end