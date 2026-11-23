
% Check smoothed gait against the original valid joint limits

raw = readmatrix('data/nn_experiment_gait.csv');
smoothed = readmatrix('data/nn_smoothed_gait.csv');

assert(isequal(size(smoothed), [300, 24]));
assert(all(isfinite(smoothed), 'all'));

% A positive, normalised moving average cannot exceed
% the minimum or maximum of its source values.
assert(all(smoothed >= min(raw, [], 1) - 1e-10, 'all'));
assert(all(smoothed <= max(raw, [], 1) + 1e-10, 'all'));

disp('Smoothed gait is finite and stays within raw gait ranges.');
