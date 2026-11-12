angles = readmatrix('data/random_gait.csv');

for frame = 1:size(angles, 1)
    plot_spider_pose(angles(frame, :));
    title(sprintf('Frame %d / %d', frame, size(angles, 1)));
    drawnow;
    pause(0.03);
end