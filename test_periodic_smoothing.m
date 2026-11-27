% Compare raw and smoothed predictions against the GA gait.
ga = readmatrix('data/best_gait_matlab.csv');
nn = readmatrix('data/nn_final_gait.csv');

assert(isequal(size(ga), [300, 24]));
assert(isequal(size(nn), [300, 24]));

assert(all(isfinite(ga), 'all'));
assert(all(isfinite(nn), 'all'));

% The 11-frame average wraps across the ends of the cycle.
radius = 5;
smoothed = zeros(size(nn));

for offset = -radius:radius
    smoothed = smoothed + circshift(nn, offset, 1);
end

smoothed = smoothed / (2 * radius + 1);

% Include the last-to-first transition in both differences.
ga_velocity = diff([ga; ga(1, :)], 1, 1);
nn_velocity = diff([nn; nn(1, :)], 1, 1);
smooth_velocity = diff([smoothed; smoothed(1, :)], 1, 1);

ga_acc = diff([ga_velocity; ga_velocity(1, :)], 1, 1);
nn_acc = diff([nn_velocity; nn_velocity(1, :)], 1, 1);
smooth_acc = diff([smooth_velocity; smooth_velocity(1, :)], 1, 1);

ga_roughness = mean(abs(ga_acc), 'all');
nn_roughness = mean(abs(nn_acc), 'all');
smooth_roughness = mean(abs(smooth_acc), 'all');

nn_mse = mean((nn - ga).^2, 'all');
smooth_mse = mean((smoothed - ga).^2, 'all');

fprintf('GA roughness: %.8f\n', ga_roughness);
fprintf('Raw NN roughness: %.8f\n', nn_roughness);
fprintf('Smoothed NN roughness: %.8f\n', smooth_roughness);

fprintf('\nRaw NN MSE: %.8f\n', nn_mse);
fprintf('Smoothed NN MSE: %.8f\n', smooth_mse);

fprintf('\nRoughness reduction: %.2f%%\n', ...
    100 * (1 - smooth_roughness / nn_roughness));

writematrix(smoothed, 'data/nn_final_smoothed_gait.csv');
