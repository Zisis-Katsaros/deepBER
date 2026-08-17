function plot_pda_eye(Ts, s1_pred_raw, s0_pred_raw, metrics_pred_raw, s1_act, s0_act, metrics_act, ...
                      mask_height, mask_width, plot_title, s1_pred_adj, s0_pred_adj, metrics_pred_adj)
    arguments
        Ts (1,1) double
        s1_pred_raw (:,1) double
        s0_pred_raw (:,1) double
        metrics_pred_raw (1,1) struct
        s1_act (:,1) double
        s0_act (:,1) double
        metrics_act (1,1) struct
        mask_height (1,1) double
        mask_width (1,1) double
        plot_title (1,1) string = "PDA Worst-Case Eye Diagram"
        s1_pred_adj = []
        s0_pred_adj = []
        metrics_pred_adj = []
    end

    % Visually Center the Eye based on the Actual Signal
    [~, max_idx] = max(s1_act - s0_act);
    center_target = round(length(s1_act) / 2);
    shift_amount = center_target - max_idx;

    s1_act_shift = circshift(s1_act, shift_amount);
    s0_act_shift = circshift(s0_act, shift_amount);
    s1_pred_raw_shift = circshift(s1_pred_raw, shift_amount);
    s0_pred_raw_shift = circshift(s0_pred_raw, shift_amount);
    
    if ~isempty(s1_pred_adj)
        s1_pred_adj_shift = circshift(s1_pred_adj, shift_amount);
        s0_pred_adj_shift = circshift(s0_pred_adj, shift_amount);
    end

    % Create time vector in picoseconds (ps)
    t_ps = (0:(length(s1_act)-1)) * Ts * 1e12; 

    % Setup the Figure
    figure('Name', 'PDA Worst-Case Eye', 'Color', 'w');
    hold on; grid on;

    % Plot Actual Bounds (Solid Red lines)
    plot(t_ps, s1_act_shift, 'r-', 'LineWidth', 2, 'DisplayName', 'Actual s_1 (Worst 1)');
    plot(t_ps, s0_act_shift, 'r-', 'LineWidth', 2, 'DisplayName', 'Actual s_0 (Worst 0)');
    
    % Plot Predicted Bounds
    if ~isempty(s1_pred_adj)
        plot(t_ps, s1_pred_raw_shift, ':', 'Color', [0.4, 0.4, 1.0], 'LineWidth', 1.5, 'DisplayName', 'Predicted Raw s_1');
        plot(t_ps, s0_pred_raw_shift, ':', 'Color', [0.4, 0.4, 1.0], 'LineWidth', 1.5, 'DisplayName', 'Predicted Raw s_0');
        
        plot(t_ps, s1_pred_adj_shift, '--', 'Color', [0, 0.5, 0], 'LineWidth', 1.5, 'DisplayName', 'Predicted Adj s_1');
        plot(t_ps, s0_pred_adj_shift, '--', 'Color', [0, 0.5, 0], 'LineWidth', 1.5, 'DisplayName', 'Predicted Adj s_0');
    else
        plot(t_ps, s1_pred_raw_shift, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Predicted s_1 (Worst 1)');
        plot(t_ps, s0_pred_raw_shift, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Predicted s_0 (Worst 0)');
    end

    % Draw the UCIe Masks 
    half_w_ps = (mask_width * 1e12) / 2;
    half_h = mask_height / 2;

    % --- Actual Mask (Red) ---
    act_shift_idx = mod(metrics_act.mask_center_idx + shift_amount - 1, length(t_ps)) + 1;
    act_center_time = t_ps(act_shift_idx);
    act_cv = metrics_act.mask_center_v;
    patch([act_center_time - half_w_ps, act_center_time + half_w_ps, act_center_time + half_w_ps, act_center_time - half_w_ps], ...
          [act_cv - half_h, act_cv - half_h, act_cv + half_h, act_cv + half_h], ...
          'red', 'FaceAlpha', 0.1, 'EdgeColor', 'r', 'LineWidth', 1.5, ...
          'DisplayName', sprintf('Actual Mask (%dmV x %dps)', round(mask_height*1e3), round(mask_width*1e12)));

    % --- Predicted Raw Mask (Blue/Dotted) ---
    pred_raw_shift_idx = mod(metrics_pred_raw.mask_center_idx + shift_amount - 1, length(t_ps)) + 1;
    pred_raw_center_time = t_ps(pred_raw_shift_idx);
    pred_raw_cv = metrics_pred_raw.mask_center_v;
    patch([pred_raw_center_time - half_w_ps, pred_raw_center_time + half_w_ps, pred_raw_center_time + half_w_ps, pred_raw_center_time - half_w_ps], ...
          [pred_raw_cv - half_h, pred_raw_cv - half_h, pred_raw_cv + half_h, pred_raw_cv + half_h], ...
          'blue', 'FaceAlpha', 0.1, 'EdgeColor', 'b', 'LineStyle', ':', 'LineWidth', 1.5, ...
          'DisplayName', 'Raw Mask');

    % --- Predicted Adj Mask (Green/Dashed) ---
    if ~isempty(metrics_pred_adj)
        pred_adj_shift_idx = mod(metrics_pred_adj.mask_center_idx + shift_amount - 1, length(t_ps)) + 1;
        pred_adj_center_time = t_ps(pred_adj_shift_idx);
        pred_adj_cv = metrics_pred_adj.mask_center_v;
        patch([pred_adj_center_time - half_w_ps, pred_adj_center_time + half_w_ps, pred_adj_center_time + half_w_ps, pred_adj_center_time - half_w_ps], ...
              [pred_adj_cv - half_h, pred_adj_cv - half_h, pred_adj_cv + half_h, pred_adj_cv + half_h], ...
              'green', 'FaceAlpha', 0.1, 'EdgeColor', [0, 0.5, 0], 'LineStyle', '--', 'LineWidth', 1.5, ...
              'DisplayName', 'Adj Mask');
    end

    % Format the Plot
    title(plot_title);
    xlabel('Time (ps)');
    ylabel('Voltage (V)');
    legend('Location', 'best');
    xlim([min(t_ps) max(t_ps)]);
    hold off;
end