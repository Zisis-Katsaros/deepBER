function visualize_fsv(ADM_mat, FDM_mat, GDM_mat, geom_idx)
    % Create a single figure wide enough to hold all three 3D plots
    figure('Name', sprintf('FSV 3D Matrix - Geometry %d', geom_idx), ...
           'Position', [100, 100, 1500, 450]);
       
    metrics = {ADM_mat, FDM_mat, GDM_mat};
    metric_names = {'ADM', 'FDM', 'GDM'};
    
    for plt_idx = 1:3
        subplot(1, 3, plt_idx);
        h = bar3(metrics{plt_idx});
        
        % Color the bars by height (Z-data) to match the reference image
        for k = 1:length(h)
            zdata = h(k).ZData;
            h(k).CData = zdata;
            h(k).FaceColor = 'interp';
        end
        
        % Formatting
        colormap('parula'); 
        colorbar;
        title(sprintf('%s of S-Matrix Geom %d', metric_names{plt_idx}, geom_idx));
        xlabel('i');
        ylabel('j');
        zlabel(sprintf('%s(S_{ij})', metric_names{plt_idx}));
        
        % Adjust axes for better viewing of multi-port matrices
        set(gca, 'XTick', 1:size(ADM_mat,2), 'YTick', 1:size(ADM_mat,1));
        set(gca, 'YDir', 'reverse'); % Reverse Y-axis for better visualization
        view(45, 30);
    end
end