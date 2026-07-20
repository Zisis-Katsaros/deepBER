addpath('transient');

start_geom = 1;
max_geoms = 2;
for geom_idx = start_geom:(start_geom + max_geoms - 1)
    filename_preds = ['out_files/pi_stcnn/touchstone_files_3/preds/geom', num2str(geom_idx), '_pred.s18p']; %['csv_files/export4transient/preds/geom', num2str(geom_idx), '_pred.s18p'];
    filename_actuals = ['out_files/pi_stcnn/touchstone_files_3/actuals/geom', num2str(geom_idx), '_actual.s18p']; %['csv_files/export4transient/actuals/geom', num2str(geom_idx), '_actual.s18p'];
    run_transient_evaluation(filename_preds, filename_actuals, sprintf('Geometry %d', geom_idx));
end