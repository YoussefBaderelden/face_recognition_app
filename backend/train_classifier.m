function train_classifier()
disp('Starting classifier training...');

% Get the faces directory
backend_dir = pwd;
faces_dir = fullfile(backend_dir, 'Faces');

% Verify faces directory exists
if ~exist(faces_dir, 'dir')
error('Faces directory not found: %s', faces_dir);
end

% Check if there are any subdirectories (person folders) - Custom implementation
entries = dir(faces_dir);
subdirs = {};
count = 0;
for i = 1:length(entries)
if entries(i).isdir && ~strcmp(entries(i).name, '.') && ~strcmp(entries(i).name, '..')
count = count + 1;
subdirs{count} = entries(i).name;
end
        end

if count == 0
error('No person folders found in Faces directory');
end

disp(['Found ', num2str(count), ' person folders']);

% Custom implementation of imageDatastore functionality
        images = {};
labels = {};
imageCount = 0;

% Loop through person folders
for i = 1:length(subdirs)
        person_folder = fullfile(faces_dir, subdirs{i});
image_files = dir(fullfile(person_folder, '*.jpg'));
image_files = [image_files; dir(fullfile(person_folder, '*.png'))];
image_files = [image_files; dir(fullfile(person_folder, '*.jpeg'))];

for j = 1:length(image_files)
        imageCount = imageCount + 1;
                images{imageCount} = fullfile(person_folder, image_files(j).name);
labels{imageCount} = subdirs{i};
end
        end

disp(['Total images found: ', num2str(imageCount)]);

if imageCount == 0
error('No images found in Faces directory');
end

% Extract features from images
imgSize = [100 100];
features = [];
final_labels = {};

disp('Extracting HOG features from images...');

for i = 1:imageCount
        try
% Read and process image - custom implementation
img = custom_imread(images{i});
img = custom_resize(custom_rgb2gray(img), imgSize);
hog = custom_extractHOGFeatures(img);
features = [features; hog];
final_labels{end+1} = labels{i};

if mod(i, 10) == 0 || i == imageCount
disp(['Processed ', num2str(i), ' of ', num2str(imageCount), ' images']);
end
        catch ME
disp(['Error processing image ', num2str(i), ': ', ME.message]);
end
        end

% Train classifier - using custom implementation
disp('Training classifier...');
try
        classifier = custom_train_multiclass_svm(features, final_labels);
save('faceClassifier.mat', 'classifier');
disp('Classifier trained and saved successfully.');
catch ME
error('Error training classifier: %s', ME.message);
end
        end

% Custom image reading function
function img = custom_imread(filepath)
fileID = fopen(filepath, 'rb');
if fileID == -1
error('Cannot open image file');
end

% For simplicity, we'll still use imread as creating a full image decoder is complex
% In a real implementation, this would be replaced with custom JPEG/PNG decoding
img = imread(filepath);
fclose(fileID);
end

% Custom RGB to grayscale conversion
        function gray = custom_rgb2gray(rgb)
if size(rgb, 3) == 3
% Standard RGB to grayscale formula: 0.2989 * R + 0.5870 * G + 0.1140 * B
        gray = 0.2989 * double(rgb(:,:,1)) + 0.5870 * double(rgb(:,:,2)) + 0.1140 * double(rgb(:,:,3));
gray = uint8(gray);
else
gray = rgb; % Already grayscale
end
        end

% Custom image resize function
function resized = custom_resize(img, new_size)
                   [orig_height, orig_width] = size(img);
resized = zeros(new_size(1), new_size(2), 'uint8');

for y = 1:new_size(1)
for x = 1:new_size(2)
% Map coordinates from target to source image
        src_x = ((x-1) * orig_width / new_size(2)) + 1;
src_y = ((y-1) * orig_height / new_size(1)) + 1;

% Simple nearest neighbor interpolation
x_idx = round(src_x);
y_idx = round(src_y);

% Boundary check
x_idx = max(1, min(orig_width, x_idx));
y_idx = max(1, min(orig_height, y_idx));

resized(y, x) = img(y_idx, x_idx);
end
        end
end

% Custom HOG feature extraction
function hog_features = custom_extractHOGFeatures(img)
                        % Simple HOG implementation
% Parameters
        cell_size = 8;
block_size = 2;
num_bins = 9;

[height, width] = size(img);

% Calculate gradients
gradX = zeros(height, width);
gradY = zeros(height, width);

% Compute gradients (central difference)
for y = 2:height-1
for x = 2:width-1
gradX(y, x) = double(img(y, x+1)) - double(img(y, x-1));
gradY(y, x) = double(img(y+1, x)) - double(img(y-1, x));
end
        end

% Compute magnitude and orientation
magnitude = sqrt(gradX.^2 + gradY.^2);
orientation = atan2(gradY, gradX) * 180 / pi;
orientation(orientation < 0) = orientation(orientation < 0) + 180;

% Compute histograms in cells
cells_y = floor(height / cell_size);
cells_x = floor(width / cell_size);
histograms = zeros(cells_y, cells_x, num_bins);

for cell_y = 1:cells_y
for cell_x = 1:cells_x
for y = 1:cell_size
for x = 1:cell_size
        img_y = (cell_y-1)*cell_size + y;
img_x = (cell_x-1)*cell_size + x;

if img_y <= height && img_x <= width
        orient = orientation(img_y, img_x);
mag = magnitude(img_y, img_x);

% Determine histogram bin
        bin = min(floor(orient / (180/num_bins)) + 1, num_bins);
histograms(cell_y, cell_x, bin) = histograms(cell_y, cell_x, bin) + mag;
end
        end
end
        end
end

% Block normalization
blocks_y = cells_y - block_size + 1;
blocks_x = cells_x - block_size + 1;
features_per_block = block_size * block_size * num_bins;

hog_features = zeros(1, blocks_y * blocks_x * features_per_block);
feature_idx = 1;

for block_y = 1:blocks_y
for block_x = 1:blocks_x
        block_features = [];
for cell_y = block_y:block_y+block_size-1
for cell_x = block_x:block_x+block_size-1
block_features = [block_features, reshape(histograms(cell_y, cell_x, :), 1, [])];
end
        end

% Normalize block
block_norm = sqrt(sum(block_features.^2) + 1e-6);
block_features = block_features / block_norm;

% Add to feature vector
hog_features(feature_idx:feature_idx+features_per_block-1) = block_features;
feature_idx = feature_idx + features_per_block;
end
        end
end

% Custom multi-class SVM training (one-vs-all approach)
function model = custom_train_multiclass_svm(features, labels)
                 % Get unique class labels
        unique_labels = unique(labels);
num_classes = length(unique_labels);

% Create structure to hold the model
model = struct();
model.classes = unique_labels;
model.binary_models = cell(1, num_classes);

% Train one-vs-all binary classifiers
for i = 1:num_classes
% Create binary labels (1 for current class, -1 for others)
current_class = unique_labels{i};
binary_labels = ones(length(labels), 1) * -1;

for j = 1:length(labels)
if strcmp(labels{j}, current_class)
binary_labels(j) = 1;
end
        end

% Train binary SVM using simplified algorithm
model.binary_models{i} = custom_train_binary_svm(features, binary_labels);
end

% Add predict function
        model.predict = @custom_predict;
end

% Custom binary SVM training (simplified version)
function binary_model = custom_train_binary_svm(X, y)
                        % Simplified SVM implementation
[num_samples, num_features] = size(X);

% Initialize weights
w = zeros(1, num_features);
b = 0;
learning_rate = 0.01;
lambda = 0.01;  % Regularization parameter
max_epochs = 100;

% Gradient descent optimization
for epoch = 1:max_epochs
for i = 1:num_samples
% Hinge loss gradient
if y(i) * (dot(w, X(i,:)) + b) < 1
dw = lambda * w - y(i) * X(i,:);
db = -y(i);
else
dw = lambda * w;
db = 0;
end

% Update weights
w = w - learning_rate * dw;
b = b - learning_rate * db;
end
        end

binary_model = struct('w', w, 'b', b);
end

% Custom prediction function for multiclass model
function predicted_label = custom_predict(model, X)
num_classes = length(model.classes);
scores = zeros(1, num_classes);

% Get scores from all binary classifiers
for i = 1:num_classes
binary_model = model.binary_models{i};
scores(i) = dot(binary_model.w, X) + binary_model.b;
end

% Find class with highest score
[~, max_idx] = max(scores);
predicted_label = model.classes{max_idx};
end