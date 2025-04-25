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

% Read the image
        try
img = imread(imagePath);
disp('Image read successfully');
catch ME
disp(['Error reading image: ', ME.message]);
return;
end

% Resize image
img = imresize(img, [300 300]);

% Face detector
faceDetector = vision.CascadeObjectDetector();
bboxes = step(faceDetector, img);

disp(['Number of faces detected: ', num2str(size(bboxes,1))]);

if ~isempty(bboxes)
% If classifier exists, use it to recognize faces
if ~isempty(classifier)
        imgSize = [100 100];
for i = 1:size(bboxes,1)
face = imcrop(img, bboxes(i,:));
faceGray = imresize(rgb2gray(face), imgSize);
hog = extractHOGFeatures(faceGray);
try
        predictedLabel = predict(classifier, hog);
label = char(predictedLabel);
disp(['Face ', num2str(i), ' classified as: ', label]);
catch ME
disp(['Error classifying face: ', ME.message]);
label = 'Unknown';
end
        img = insertObjectAnnotation(img, 'rectangle', bboxes(i,:), label);
end
else
disp('No classifier available, only marking face rectangles');
img = insertShape(img, 'Rectangle', bboxes, 'LineWidth', 3, 'Color', 'green');
end
else
disp('No faces detected in the image');
end

% Save the output image
save_output_image(imagePath, img);
disp('Processing completed');
end