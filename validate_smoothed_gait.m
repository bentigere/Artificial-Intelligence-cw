% Check the dimensions and joint limits after smoothing.

raw = readmatrix('data/nn_final_gait.csv');
smoothed = readmatrix('data/nn_final_smoothed_gait.csv');

assert(isequal(size(raw), [300, 24]));
assert(isequal(size(smoothed), [300, 24]));
assert(all(isfinite(smoothed), 'all'));

lower_limits = repmat([-1.5708, -1.5708, -2.0], 1, 8);
upper_limits = repmat([1.5708, 1.5708, 1.0], 1, 8);

assert(all(smoothed >= lower_limits & ...
    smoothed <= upper_limits, 'all'), ...
    'Smoothed gait violates joint limits.');

disp('Final smoothed NN gait passed validation.');
