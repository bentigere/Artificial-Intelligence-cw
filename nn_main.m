clear;
clc;

% Select the NN output, using the same choices as main_nn.m.
project_dir = fileparts(mfilename('fullpath'));
addpath(project_dir);
nn_gait_type = "raw"; % "raw" or "smoothed"

switch nn_gait_type
    case "raw"
        nn_file = 'nn_predicted_gait.csv';
    case "smoothed"
        nn_file = 'nn_smoothed_gait.csv';
    otherwise
        error('Unknown Neural Network gait type.');
end

angles = readmatrix(fullfile(project_dir, 'data', nn_file));
assert(isequal(size(angles), [300, 24]), 'Expected 300 rows and 24 angles.');
assert(all(isfinite(angles(:))), 'Predicted angles must be finite.');
nn_lower = repmat([-1.5708, -1.5708, -2.0], 1, 8);
nn_upper = repmat([1.5708, 1.5708, 1.0], 1, 8);
assert(all(all(angles >= nn_lower & angles <= nn_upper)), ...
    'Predicted angles violate the repository joint limits.');

num_frames = size(angles, 1);
repetitions = 10;
cycle_duration = 1.55; % Target seconds per complete gait cycle.
frame_duration = cycle_duration / num_frames;

% Spider geometry, matching plot_spider_pose.m.
segment_lengths = [1.2, 0.7, 1.0];
base_angles = deg2rad([45, 75, 105, 135, -135, -105, -75, -45]);
leg_names = {'L1', 'L2', 'L3', 'L4', 'R4', 'R3', 'R2', 'R1'};

% Create the figure only once.
fig = figure('Color', 'w', 'Name', 'NN Predicted Spider Gait');
ax = axes('Parent', fig, 'Color', 'w', ...
    'XColor', 'k', 'YColor', 'k', 'ZColor', 'k');
hold(ax, 'on');
grid(ax, 'on');
axis(ax, 'equal');
xlim(ax, [-4, 4]);
ylim(ax, [-4, 4]);
zlim(ax, [-2, 2]);
axis(ax, 'manual');
view(ax, 45, 45);
xlabel(ax, 'X');
ylabel(ax, 'Y');
zlabel(ax, 'Z');

% Draw the fixed body and head.
t = linspace(0, 2*pi, 100);
plot3(ax, 1.5*cos(t), sin(t), zeros(size(t)), 'k-', 'LineWidth', 3);
plot3(ax, 1.7, 0, 0, 'r^', 'MarkerFaceColor', 'r');

% Create each leg once. Its coordinates will be updated below.
leg_lines = gobjects(8, 1);
base_positions = zeros(8, 3);

for leg = 1:8
    a = base_angles(leg);
    base_positions(leg, :) = [1.5*cos(a), sin(a), 0];

    leg_lines(leg) = plot3(ax, nan, nan, nan, 'b-o', ...
        'LineWidth', 2, 'MarkerSize', 4);

    label_pos = base_positions(leg, :) + ...
        [0.2*cos(a), 0.2*sin(a), 0.05];

    text(ax, label_pos(1), label_pos(2), label_pos(3), ...
        leg_names{leg}, 'FontWeight', 'bold', 'Color', 'k');
end

title_handle = title(ax, 'NN predicted gait', 'Color', 'k');
drawnow;

% Repeat the same gait without clearing or recreating the figure.
for cycle = 1:repetitions
    cycle_start = tic;

    for frame = 1:num_frames
        if ~isgraphics(fig)
            return; % Stop if the window is closed.
        end

        for leg = 1:8
            columns = (leg - 1)*3 + (1:3);

            [j1, j2, j3, j4] = forward_leg_kinematics2( ...
                base_positions(leg, :), ...
                base_angles(leg), ...
                angles(frame, columns), ...
                segment_lengths);

            points = [j1(:)'; j2(:)'; j3(:)'; j4(:)'];

            set(leg_lines(leg), ...
                'XData', points(:, 1), ...
                'YData', points(:, 2), ...
                'ZData', points(:, 3));
        end

        set(title_handle, 'String', sprintf( ...
            'NN %s gait - Cycle %d / %d - Frame %d / %d', ...
            char(nn_gait_type), cycle, repetitions, frame, num_frames));

        drawnow;

        % Account for time already spent calculating and drawing.
        remaining = frame * frame_duration - toc(cycle_start);
        if remaining > 0
            pause(remaining);
        end
    end
end
