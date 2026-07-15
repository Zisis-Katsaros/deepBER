addpath('transient');

start_geom = 1;
max_geoms = 2;
for geom_idx = start_geom:max_geoms
    filename_preds = ['csv_files/export4transient/preds/geom', num2str(geom_idx), '_pred.s18p'];
    filename_actuals = ['csv_files/export4transient/actuals/geom', num2str(geom_idx), '_actual.s18p'];
    run_transient_evaluation(filename_preds, filename_actuals, sprintf('Geometry %d', geom_idx));
end