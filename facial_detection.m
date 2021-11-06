%% facial_detection.m
% Cooper Yancey

%% Setup

% Clean up
close all; clear all; clc;

% Reading image
[file,path]   = uigetfile('*.jpg;*.png;','Select an image file');
selected_file = fullfile(path,file);
img           = imread(selected_file);
imshow(img); title('Original Image');

%% Viola-Jones Face Detector

% Initialize detector
vj_face_detector  = vision.CascadeObjectDetector;
vj_face_detector.MergeThreshold = 7;

% Create shape inserter and define color as blue
shape_inserter = vision.ShapeInserter('BorderColor', 'Custom', 'CustomBorderColor', [0 0 255]);

% Detecting faces and putting in bbox (doesn't detect askew faces)
bboxes = step(vj_face_detector, img);

% Drawing boxes and showing faces (if detected)
if ~isempty(bboxes)
    img_faces = step(shape_inserter, img, int32(bboxes));
    figure; imshow(img_faces); title('Faces Detected in Image with Viola-Jones Algorithm');
else
    img_out = insertText(img, [0 0], 'No Faces Detected with VJ!', 'fontsize', 15, 'BoxOpacity', 1);
    figure; imshow(img_out); title('Viola-Jones Algorithm');
end

%% Multi-Task Cascaded Convolutional Neural Networks (MTCNNs) Face Detector

% Get values from the face detector
[bbox, scores, landmarks] = mtcnn.detectFaces(img);

% Drawing the boxes and showing faces (if detected)
if ~isempty(bbox)
    img_out = insertObjectAnnotation(img, 'rectangle', bbox, scores, 'LineWidth', 2);
    figure; imshow(img_out), title('Faces Detected in Image with Confidence and Features Shown with MTCNN');
    hold on;
    for iFace = 1:numel(scores)
        scatter(landmarks(iFace, :, 1), landmarks(iFace, :, 2), 'filled');
    end

else
    img_out = insertText(img, [0 0], 'No Faces Detected with MTCNN!', 'fontsize', 15, 'BoxOpacity', 1);
    figure; imshow(img_out); title('MTCNN Algorithm');
end
