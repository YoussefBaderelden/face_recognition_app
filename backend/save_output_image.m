function save_output_image(originalPath, img)
% Get the backend directory (current working directory)
        backend_dir = pwd;

% Define results folder path
result_folder = fullfile(backend_dir, 'results');

% Ensure results directory exists
if ~exist(result_folder, 'dir')
mkdir(result_folder);
end

% Extract filename without full path
[~, name, ext] = fileparts(originalPath);

% Create output filename
        outName = fullfile(result_folder, [name, '_output', ext]);

% Save the processed image
imwrite(img, outName);

% Display path for debugging
disp(['Saved processed image to: ', outName]);
end