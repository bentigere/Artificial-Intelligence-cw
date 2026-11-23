% NN playback companion to main.m. The lecturer's plotting/FK functions are reused.
% From MATLAB, change Current Folder to the repository root and run main_nn.
nn_root = fileparts(mfilename('fullpath'));
addpath(nn_root);

% Select the Neural Network gait to visualise
nn_gait_type = "raw";

switch nn_gait_type
    case "raw"
        nn_file = 'nn_experiment_gait.csv';

    case "smoothed"
        nn_file = 'nn_smoothed_gait.csv';

    otherwise
        error('Unknown Neural Network gait type.');
end

nn_angles = readmatrix(fullfile(nn_root, 'data', nn_file));
assert(isequal(size(nn_angles), [300, 24]), 'Expected 300 rows and 24 angles.');
assert(all(isfinite(nn_angles(:))), 'Predicted angles must be finite.');
nn_lower = repmat([-1.5708, -1.5708, -2.0], 1, 8);
nn_upper = repmat([1.5708, 1.5708, 1.0], 1, 8);
assert(all(all(nn_angles >= nn_lower & nn_angles <= nn_upper)), ...
    'Predicted angles violate the repository joint limits.');

nn_images = fullfile(nn_root, 'images', 'nn', char(nn_gait_type));
if ~exist(nn_images, 'dir')
    mkdir(nn_images);
end

for frame = 1:size(nn_angles, 1)
    plot_spider_pose(nn_angles(frame, :));
    % Keep the lecturer's black body/leg lines visible in MATLAB's dark theme.
    set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'ZColor', 'k');
    set(findall(gca, 'Type', 'text'), 'Color', 'k');
    title(sprintf('NN predicted gait - Frame %d / %d', frame, size(nn_angles, 1)), 'Color', 'k');
    drawnow;
    if ismember(frame, [1, 75, 150, 225, 300])
        exportgraphics(gcf, fullfile(nn_images, sprintf('matlab_nn_frame_%03d.png', frame)));
    end
    pause(0.03);
end
