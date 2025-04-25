function train_classifier()
disp('Starting classifier training...');

% Get the faces directory
backend_dir = pwd;
faces_dir = fullfile(backend_dir, 'Faces');

% Verify faces directory exists
if ~exist(faces_dir, 'dir')
error('Faces directory not found: %s', faces_dir);
end

% Check if there are any subdirectories (person folders)
        subdirs = dir(faces_dir);
        subdirs = subdirs([subdirs.isdir]);
        subdirs = subdirs(~ismember({subdirs.name}, {'.', '..'}));

if isempty(subdirs)
error('No person folders found in Faces directory');
end

disp(['Found ', num2str(length(subdirs)), ' person folders']);

% Load image datastore
        try
faceData = imageDatastore(faces_dir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
disp(['Total images found: ', num2str(numel(faceData.Files))]);
catch ME
error('Error loading image datastore: %s', ME.message);
end

if numel(faceData.Files) == 0
error('No images found in Faces directory');
end

% Extract features from images
imgSize = [100 100];
features = [];
labels = [];

disp('Extracting HOG features from images...');

for i = 1:numel(faceData.Files)
try
        img = readimage(faceData, i);
img = imresize(rgb2gray(img), imgSize);
hog = extractHOGFeatures(img);
features = [features; hog];
labels = [labels; faceData.Labels(i)];

if mod(i, 10) == 0 || i == numel(faceData.Files)
disp(['Processed ', num2str(i), ' of ', num2str(numel(faceData.Files)), ' images']);
end
        catch ME
disp(['Error processing image ', num2str(i), ': ', ME.message]);
end
        end

% Train classifier
disp('Training classifier...');
try
        classifier = fitcecoc(features, labels);
save('faceClassifier.mat', 'classifier');
disp('Classifier trained and saved successfully.');
catch ME
error('Error training classifier: %s', ME.message);
end
        end