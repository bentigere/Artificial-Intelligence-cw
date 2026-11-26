
% Compare GA gait and Neural Network gait smoothness

ga = readmatrix('data/best_gait_matlab.csv');
nn = readmatrix('data/nn_final_gait.csv');

% Ensure both gaits have 300 frames and 24 joint angles
assert(isequal(size(ga), [300, 24]));
assert(isequal(size(nn), [300, 24]));

% Include the transition from the final frame
% back to the first frame for periodic walking.
ga_velocity = diff([ga; ga(1, :)], 1, 1);
nn_velocity = diff([nn; nn(1, :)], 1, 1);

% Second differences measure changes in movement
ga_acceleration = diff([ga_velocity; ga_velocity(1, :)], 1, 1);
nn_acceleration = diff([nn_velocity; nn_velocity(1, :)], 1, 1);

% Average absolute angular change per frame
ga_speed = mean(abs(ga_velocity), 'all');
nn_speed = mean(abs(nn_velocity), 'all');

% Average absolute second difference
ga_roughness = mean(abs(ga_acceleration), 'all');
nn_roughness = mean(abs(nn_acceleration), 'all');

% Prediction accuracy
mse = mean((nn - ga).^2, 'all');

fprintf('GA average frame change: %.6f rad\n', ga_speed);
fprintf('NN average frame change: %.6f rad\n', nn_speed);

fprintf('GA roughness: %.6f rad\n', ga_roughness);
fprintf('NN roughness: %.6f rad\n', nn_roughness);

fprintf('Prediction MSE: %.8f rad^2\n', mse);
