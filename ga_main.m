clear;
clc;

% Load the generated gait: 300 frames × 24 joint angles.
project_dir = fileparts(mfilename('fullpath'));
angles = readmatrix(fullfile(project_dir, 'data', 'best_gait_matlab.csv'));

assert(size(angles, 2) == 24, 'Each frame must contain 24 angles.');

num_frames = size(angles, 1);
repetitions = 10;
cycle_duration = 1.55; % Target seconds per complete gait cycle.
frame_duration = cycle_duration / num_frames;

% Spider geometry, matching plot_spider_pose.m.
segment_lengths = [1.2, 0.7, 1.0];
base_angles = deg2rad([45, 75, 105, 135, -135, -105, -75, -45]);
leg_names = {'L1', 'L2', 'L3', 'L4', 'R4', 'R3', 'R2', 'R1'};

% MATLAB reference standing pose
reference_angles = [0, -pi/4, -pi/3];

% Calculate the ground height from the reference pose
reference_foot_heights = zeros(1, 8);

for leg = 1:8
    angle = base_angles(leg);

    base_pos = [
        1.5 * cos(angle), ...
        1.0 * sin(angle), ...
        0
    ];

    [~, ~, ~, foot] = forward_leg_kinematics2( ...
        base_pos, angle, reference_angles, segment_lengths);

    reference_foot_heights(leg) = foot(3);
end

ground_z = mean(reference_foot_heights);

fprintf('Ground height: %.4f\n', ground_z);


% Create the figure only once.
fig = figure('Color', 'w', 'Name', 'Generated Spider Gait');
ax = axes('Parent', fig);
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
        leg_names{leg}, 'FontWeight', 'bold');
end

title_handle = title(ax, 'Generated gait');
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
            'Cycle %d / %d — Frame %d / %d', ...
            cycle, repetitions, frame, num_frames));

        drawnow;

        % Account for time already spent calculating and drawing.
        remaining = frame * frame_duration - toc(cycle_start);
        if remaining > 0
            pause(remaining);
        end
    end
end