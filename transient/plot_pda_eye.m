function plot_pda_eye(Ts, s1_pred, s0_pred, s1_act, s0_act, metrics_pred, metrics_act, mask_height, mask_width, plot_title)
    %{
    Plots the worst-case eye opening calculated by Peak Distortion Analysis.
    Overlays two optimally placed UCIe specification masks (Actual and Predicted).
    %}
    arguments
        Ts (1,1) double
        s1_pred (:,1) double
        s0_pred (:,1) double
        s1_act (:,1) double
        s0_act (:,1) double
        metrics_pred (1,1) struct
        metrics_act (1,1) struct
        mask_height (1,1) double
        mask_width (1,1) double
        plot_title (1,1) string = "PDA Worst-Case Eye Diagram"
    end

    % 1. Visually Center the Eye based on the Actual Signal
    [~, max_idx] = max(s1_act - s0_act);
    center_target = round(length(s1_act) / 2);
    shift_amount = center_target - max_idx;

    s1_act_shift = circshift(s1_act, shift_amount);
    s0_act_shift = circshift(s0_act, shift_amount);
    s1_pred_shift = circshift(s1_pred, shift_amount);
    s0_pred_shift = circshift(s0_pred, shift_amount);

    % Create time vector in picoseconds (ps)
    t_ps = (0:(length(s1_act)-1)) * Ts * 1e12; 

    % 2. Setup the Figure
    figure('Name', 'PDA Worst-Case Eye', 'Color', 'w');
    hold on; grid on;

    % Plot Predicted Bounds (Dashed Blue lines)
    plot(t_ps, s1_pred_shift, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Predicted s_1 (Worst 1)');
    plot(t_ps, s0_pred_shift, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Predicted s_0 (Worst 0)');

    % Plot Actual Bounds (Solid Red lines)
    plot(t_ps, s1_act_shift, 'r-', 'LineWidth', 2, 'DisplayName', 'Actual s_1 (Worst 1)');
    plot(t_ps, s0_act_shift, 'r-', 'LineWidth', 2, 'DisplayName', 'Actual s_0 (Worst 0)');

    % 3. Draw the UCIe Masks 
    half_w_ps = (mask_width * 1e12) / 2;
    half_h = mask_height / 2;

    % --- Actual Mask (Green) ---
    act_shift_idx = mod(metrics_act.mask_center_idx + shift_amount - 1, length(t_ps)) + 1;
    act_center_time = t_ps(act_shift_idx);
    act_cv = metrics_act.mask_center_v;
    
    patch([act_center_time - half_w_ps, act_center_time + half_w_ps, act_center_time + half_w_ps, act_center_time - half_w_ps], ...
          [act_cv - half_h, act_cv - half_h, act_cv + half_h, act_cv + half_h], ...
          'green', 'FaceAlpha', 0.25, 'EdgeColor', 'k', 'LineWidth', 1.5, ...
          'DisplayName', sprintf('Actual Mask (%dmV x %dps)', round(mask_height*1e3), round(mask_width*1e12)));

    % --- Predicted Mask (Blue/Dashed) ---
    pred_shift_idx = mod(metrics_pred.mask_center_idx + shift_amount - 1, length(t_ps)) + 1;
    pred_center_time = t_ps(pred_shift_idx);
    pred_cv = metrics_pred.mask_center_v;

    patch([pred_center_time - half_w_ps, pred_center_time + half_w_ps, pred_center_time + half_w_ps, pred_center_time - half_w_ps], ...
          [pred_cv - half_h, pred_cv - half_h, pred_cv + half_h, pred_cv + half_h], ...
          'blue', 'FaceAlpha', 0.15, 'EdgeColor', 'b', 'LineStyle', '--', 'LineWidth', 1.5, ...
          'DisplayName', sprintf('Predicted Mask (%dmV x %dps)', round(mask_height*1e3), round(mask_width*1e12)));

    % 4. Format the Plot
    title(plot_title);
    xlabel('Time (ps)');
    ylabel('Voltage (V)');
    legend('Location', 'best');
    xlim([min(t_ps) max(t_ps)]);
    hold off;
end