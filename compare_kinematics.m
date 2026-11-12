% test if the converted python file generates exactly the same output as the original forward_leg_kinematics2.m

% Load the same angles used by Python and its calculated coordinates.
folder = fileparts(mfilename('fullpath'));
data_folder = fullfile(folder, 'test', 'test_data');

angles = readmatrix(fullfile(data_folder, 'test_angles.csv'));
expected = readmatrix(fullfile(data_folder, 'python_coordinates.csv'));

assert(isequal(size(angles), [300, 24]));

addpath(fileparts(folder));

% Same geometry and leg order as plot_spider_pose.m.
base_angles = deg2rad([45, 75, 105, 135, -135, -105, -75, -45]);
segment_lengths = [1.2, 0.7, 1.0];

% Columns: frame, leg, j1 XYZ, j2 XYZ, j3 XYZ, j4 XYZ.
actual = zeros(300 * 8, 14);
row = 0;

for frame = 1:300
    for leg = 1:8
        angle = base_angles(leg);
        base_pos = [1.5*cos(angle), sin(angle), 0];

        % Select this leg's three angles.
        idx = (leg - 1)*3 + (1:3);

        [j1, j2, j3, j4] = forward_leg_kinematics2( ...
            base_pos, angle, angles(frame, idx), segment_lengths);

        row = row + 1;
        actual(row, :) = [frame, leg, j1, j2, j3, j4];
    end
end

% Save MATLAB's results.
writematrix(actual, fullfile(folder, 'matlab_coordinates.csv'));

% Check dimensions, finite values, and matching frame/leg identifiers.
assert(isequal(size(actual), size(expected)));
assert(all(isfinite(actual(:))) && all(isfinite(expected(:))));
assert(isequal(actual(:, 1:2), expected(:, 1:2)));

% Compare all XYZ coordinates.
errors = abs(actual(:, 3:14) - expected(:, 3:14));
max_error = max(errors(:));

fprintf('Compared 300 frames, 8 legs, all four points.\n');
fprintf('Maximum coordinate difference: %.12g\n', max_error);

assert(max_error < 1e-10, ...
    'FAIL: Python and MATLAB coordinates differ.');

fprintf('PASS: Coordinates match within 1e-10.\n');
writematrix(actual, fullfile(data_folder, 'matlab_coordinates.csv'));