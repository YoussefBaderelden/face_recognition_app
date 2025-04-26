function classify_image_simple(imagePath)
% Display input path for debugging
disp(['Processing image: ', imagePath]);

% Load classifier if not loaded
persistent classifier
if isempty(classifier)
try
load('faceClassifier.mat', 'classifier');
disp('Classifier loaded successfully');
catch ME
disp(['Error loading classifier: ', ME.message]);
classifier = [];
end
        end

% Read the image - using custom implementation
        try
img = custom_imread(imagePath);
disp('Image read successfully');
catch ME
disp(['Error reading image: ', ME.message]);
return;
end

% Resize image - using custom implementation
        img = custom_resize(img, [300 300]);

% Face detector - using custom implementation
        bboxes = custom_face_detector(img);

disp(['Number of faces detected: ', num2str(size(bboxes,1))]);

if ~isempty(bboxes)
% If classifier exists, use it to recognize faces
if ~isempty(classifier)
        imgSize = [100 100];
for i = 1:size(bboxes,1)
face = custom_imcrop(img, bboxes(i,:));
faceGray = custom_resize(custom_rgb2gray(face), imgSize);
hog = custom_extractHOGFeatures(faceGray);
try
        predictedLabel = classifier.predict(classifier, hog);
label = char(predictedLabel);
disp(['Face ', num2str(i), ' classified as: ', label]);
catch ME
disp(['Error classifying face: ', ME.message]);
label = 'Unknown';
end
        img = custom_insertObjectAnnotation(img, 'rectangle', bboxes(i,:), label);
end
else
disp('No classifier available, only marking face rectangles');
img = custom_insertShape(img, 'Rectangle', bboxes, 'LineWidth', 3, 'Color', 'green');
end
else
disp('No faces detected in the image');
end

% Save the output image
save_output_image(imagePath, img);
disp('Processing completed');
end

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

        function gray = custom_rgb2gray(rgb)
if size(rgb, 3) == 3
% Standard RGB to grayscale formula: 0.2989 * R + 0.5870 * G + 0.1140 * B
        gray = 0.2989 * double(rgb(:,:,1)) + 0.5870 * double(rgb(:,:,2)) + 0.1140 * double(rgb(:,:,3));
gray = uint8(gray);
else
gray = rgb; % Already grayscale
end
        end

function resized = custom_resize(img, new_size)
                   [orig_height, orig_width, channels] = size(img);

if length(new_size) ~= 2
error('New size must be a 2-element vector [height width]');
end

if numel(channels) == 0
channels = 1;
end

        resized = zeros(new_size(1), new_size(2), channels, 'uint8');

for c = 1:channels
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

if channels == 1
resized(y, x) = img(y_idx, x_idx);
else
resized(y, x, c) = img(y_idx, x_idx, c);
end
        end
end
        end
end

        function hog_features = custom_extractHOGFeatures(img)
                                % Parameters
cell_size = 8;
block_size = 2;
num_bins = 9;

[height, width] = size(img);

% Pad image to handle boundaries
        padded_img = zeros(height+2, width+2);
padded_img(2:end-1, 2:end-1) = img;

% Calculate gradients
gradX = zeros(height, width);
gradY = zeros(height, width);
for y = 1:height
for x = 1:width
gradX(y, x) = double(padded_img(y+1, x+2)) - double(padded_img(y+1, x));
gradY(y, x) = double(padded_img(y+2, x+1)) - double(padded_img(y, x+1));
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
% Improved normalization with clipping
block_norm = sqrt(sum(block_features.^2) + 1e-6);
block_features = block_features / block_norm;
block_features = min(block_features, 0.2); % Clipping
        block_norm = sqrt(sum(block_features.^2) + 1e-6);
block_features = block_features / block_norm;

hog_features(feature_idx:feature_idx+features_per_block-1) = block_features;
feature_idx = feature_idx + features_per_block;
end
        end
end

        function cropped = custom_imcrop(img, bbox)
x = max(1, round(bbox(1)));
y = max(1, round(bbox(2)));
width = round(bbox(3));
height = round(bbox(4));

[img_height, img_width, channels] = size(img);

% Ensure dimensions are within image bounds
x_end = min(img_width, x + width - 1);
y_end = min(img_height, y + height - 1);

if channels == 1
cropped = img(y:y_end, x:x_end);
else
cropped = img(y:y_end, x:x_end, :);
end
        end

function bboxes = custom_face_detector(img)
                  [rows, cols, ~] = size(img);

% Convert to grayscale for edge detection
gray = custom_rgb2gray(img);

% Simple edge detection using Sobel-like filters
edges = zeros(rows, cols);
for i = 2:rows-1
for j = 2:cols-1
gx = double(gray(i-1, j+1)) + 2*double(gray(i, j+1)) + double(gray(i+1, j+1)) - ...
double(gray(i-1, j-1)) - 2*double(gray(i, j-1)) - double(gray(i+1, j-1));
gy = double(gray(i-1, j-1)) + 2*double(gray(i-1, j)) + double(gray(i-1, j+1)) - ...
double(gray(i+1, j-1)) - 2*double(gray(i+1, j)) - double(gray(i+1, j+1));
edges(i, j) = sqrt(gx^2 + gy^2);
end
        end
edges = edges > 50; % Threshold for edges

% Skin color detection in RGB
        skin_mask = false(rows, cols);
for i = 1:rows
for j = 1:cols
        r = double(img(i, j, 1));
g = double(img(i, j, 2));
b = double(img(i, j, 3));
% Improved skin color rule
if r > 95 && g > 40 && b > 20 && r > g && r > b && abs(r - g) > 15 && r + g + b > 150
skin_mask(i, j) = true;
end
        end
end

% Combine skin mask and edges
        combined_mask = skin_mask & edges;

% Label connected components
        labeled = custom_bwlabel(combined_mask);

% Extract region properties
        regions = custom_regionprops(labeled);

% Filter regions based on size and aspect ratio
min_area = (rows * cols) * 0.01;  % At least 1% of image
max_area = (rows * cols) * 0.5;   % At most 50% of image
valid_regions = [];
for i = 1:length(regions)
if regions(i).Area > min_area && regions(i).Area < max_area
        bbox = regions(i).BoundingBox;
aspect_ratio = bbox(3) / bbox(4); % width / height
% Typical face aspect ratio is around 0.8 to 1.3
if aspect_ratio > 0.8 && aspect_ratio < 1.3
valid_regions = [valid_regions; i];
end
        end
end

% Convert region data to bounding boxes
bboxes = zeros(length(valid_regions), 4);
for i = 1:length(valid_regions)
        region_idx = valid_regions(i);
                bboxes(i, :) = regions(region_idx).BoundingBox;
end
        end

function labeled = custom_bwlabel(bw)
                   [rows, cols] = size(bw);
labeled = zeros(rows, cols);
current_label = 0;
equivalence = zeros(1000, 1); % For union-find

% First pass: assign preliminary labels and build equivalence classes
for i = 1:rows
for j = 1:cols
if bw(i, j)
        neighbors = [];
if i > 1 && labeled(i-1, j) > 0
neighbors = [neighbors, labeled(i-1, j)];
end
if j > 1 && labeled(i, j-1) > 0
neighbors = [neighbors, labeled(i, j-1)];
end
if isempty(neighbors)
        current_label = current_label + 1;
                labeled(i, j) = current_label;
equivalence(current_label) = current_label;
else
% Assign smallest label from neighbors
        min_label = min(neighbors);
labeled(i, j) = min_label;
% Union all neighbor labels
for k = 1:length(neighbors)
        root1 = find_root(equivalence, min_label);
        root2 = find_root(equivalence, neighbors(k));
if root1 ~= root2
equivalence(root2) = root1;
end
        end
end
        end
end
        end

% Second pass: resolve equivalence classes
for i = 1:rows
for j = 1:cols
if labeled(i, j) > 0
labeled(i, j) = find_root(equivalence, labeled(i, j));
end
        end
end

% Re-label components to ensure consecutive labels
unique_labels = unique(labeled(labeled > 0));
for i = 1:length(unique_labels)
labeled(labeled == unique_labels(i)) = i;
end
        end

function root = find_root(equivalence, label)
root = label;
while equivalence(root) ~= root
        root = equivalence(root);
end
% Path compression
while equivalence(label) ~= root
        next = equivalence(label);
equivalence(label) = root;
label = next;
end
        end

function regions = custom_regionprops(labeled)
max_label = max(max(labeled));
regions = struct('Area', {}, 'BoundingBox', {});

for label = 1:max_label
% Find all pixels with current label
[y, x] = find(labeled == label);

if ~isempty(y)
% Calculate area
area = length(y);

% Calculate bounding box [x, y, width, height]
min_x = min(x);
min_y = min(y);
max_x = max(x);
max_y = max(y);
width = max_x - min_x + 1;
height = max_y - min_y + 1;

regions(label).Area = area;
regions(label).BoundingBox = [min_x, min_y, width, height];
end
        end
end

        function annotated = custom_insertObjectAnnotation(img, shape, bbox, label)
annotated = img;

% Draw rectangle
annotated = custom_insertShape(annotated, shape, bbox, 'LineWidth', 2, 'Color', 'red');

% Add text label
        x = bbox(1);
y = max(1, bbox(2) - 15); % Place text above box

% Simple text rendering (rectangle with text inside)
        text_height = 15;
        text_width = length(label) * 8; % Approximation

% Draw text background
for i = y:y+text_height
for j = x:x+text_width
if i > 0 && j > 0 && i <= size(annotated, 1) && j <= size(annotated, 2)
annotated(i, j, 1) = 255; % White background
annotated(i, j, 2) = 255;
annotated(i, j, 3) = 255;
end
        end
end

% We'll still need to use text insertion from MATLAB for proper text rendering
% In a real implementation, we would render text pixel by pixel
annotated = insertText(annotated, [x, y], label, 'BoxOpacity', 0, 'TextColor', 'black');
end

        function result = custom_insertShape(img, shape, boxes, varargin)
result = img;
[height, width, ~] = size(img);

% Parse optional arguments
        lineWidth = 1;
color = [255, 0, 0]; % Default to red

for i = 1:2:length(varargin)
if strcmpi(varargin{i}, 'LineWidth')
lineWidth = varargin{i+1};
elseif strcmpi(varargin{i}, 'Color')
if ischar(varargin{i+1})
switch lower(varargin{i+1})
case 'red'
color = [255, 0, 0];
case 'green'
color = [0, 255, 0];
case 'blue'
color = [0, 0, 255];
otherwise
        color = [255, 0, 0];
end
else
color = varargin{i+1};
end
        end
end

% Convert to uint8 if necessary
if max(color) <= 1
color = uint8(color * 255);
else
color = uint8(color);
end

% Draw each box
if strcmpi(shape, 'Rectangle')
for i = 1:size(boxes, 1)
box = boxes(i, :);
x = max(1, round(box(1)));
y = max(1, round(box(2)));
w = round(box(3));
h = round(box(4));

% Draw horizontal lines
for j = max(1, x-lineWidth+1):min(width, x+w+lineWidth-1)
for l = 1:lineWidth
% Top line
if y+l-1 > 0 && y+l-1 <= height
result(y+l-1, j, 1) = color(1);
result(y+l-1, j, 2) = color(2);
result(y+l-1, j, 3) = color(3);
end
% Bottom line
if y+h-l > 0 && y+h-l <= height
result(y+h-l, j, 1) = color(1);
result(y+h-l, j, 2) = color(2);
result(y+h-l, j, 3) = color(3);
end
        end
end

% Draw vertical lines
for j = max(1, y-lineWidth+1):min(height, y+h+lineWidth-1)
for l = 1:lineWidth
% Left line
if x+l-1 > 0 && x+l-1 <= width
result(j, x+l-1, 1) = color(1);
result(j, x+l-1, 2) = color(2);
result(j, x+l-1, 3) = color(3);
end
% Right line
if x+w-l > 0 && x+w-l <= width
result(j, x+w-l, 1) = color(1);
result(j, x+w-l, 2) = color(2);
result(j, x+w-l, 3) = color(3);
end
        end
end
        end
end
        end