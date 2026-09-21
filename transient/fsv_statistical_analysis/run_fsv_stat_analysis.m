function run_fsv_stat_analysis(global_ADM, global_FDM, global_GDM, global_ADMc, global_FDMc, global_GDMc, start_geom, max_geoms)
    ADM_stats = calculate_fsv_stats(global_ADM);
    FDM_stats = calculate_fsv_stats(global_FDM);
    GDM_stats = calculate_fsv_stats(global_GDM);

    % Initialize dynamic table variables
    table_columns = {};
    var_names = {};
    Row_Metrics = {};

    % Fill table
    Row_Metrics = fieldnames(ADM_stats); 
    table_columns{end+1} = cell2mat(struct2cell(ADM_stats));
    table_columns{end+1} = cell2mat(struct2cell(FDM_stats));
    table_columns{end+1} = cell2mat(struct2cell(GDM_stats));
    var_names = [var_names, {'ADM', 'FDM', 'GDM'}];

    % DISPLAY TABLE 
    % Construct the table dynamically based on collected columns
    fsv_table = table(table_columns{:}, 'RowNames', Row_Metrics, 'VariableNames', var_names);
    disp(fsv_table);

    % Per geometry bar plots for ADM, FDM, and GDM
    plot_fsv_descriptor_bar_plot(global_ADMc, 'ADM', start_geom, max_geoms);
    plot_fsv_descriptor_bar_plot(global_FDMc, 'FDM', start_geom, max_geoms);
    plot_fsv_descriptor_bar_plot(global_GDMc, 'GDM', start_geom, max_geoms);

    plot_fsv_stats(global_ADM, 'ADM');
    plot_fsv_stats(global_FDM, 'FDM');
    plot_fsv_stats(global_GDM, 'GDM');
end
