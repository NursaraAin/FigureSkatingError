netCNN2 = resnet50;
% dataFolder=fullfile("C:\Users\nursa\OneDrive - Universiti Malaya\FigureSkatingError\");
path=fullfile("C:\Users\nursa\OneDrive - Universiti Malaya\FigureSkatingError\under-rotation\");
[files,labels] = TrainingDataFiles(path);
%%
% idx = 1;
% filename = files(idx);
% video = readVideo(filename);
% size(video)
%%
inputSize = netCNN2.Layers(1).InputSize(1:2);
layerName = "avg_pool";

tempFile = fullfile(path);

overwriteSequences = true;

if exist(tempFile,'file') && ~overwriteSequences
    load(tempFile)
else
    numFiles = numel(files);
    sequences = cell(numFiles,1);
    
    for i = 1:numFiles
        fprintf("Reading file %d of %d...\n", i, numFiles)
        
        video = readVideo(files(i));
        video = imresize(video,inputSize);
        sequences{i,1} = activations(netCNN2,video,layerName,'OutputAs','columns');
        
    end
    % Save the sequences and the labels associated with them.
    save(tempFile,"sequences","labels","-v7.3");
end
%sequences(1:10)
%%
numObservations = numel(sequences);
idx = randperm(numObservations);
N = floor(0.9 * numObservations);

idxTrain = idx(1:N);
sequencesTrain = sequences(idxTrain);
labelsTrain = labels(idxTrain);

idxValidation = idx(N+1:end);
sequencesValidation = sequences(idxValidation);
labelsValidation = labels(idxValidation);
%%
numObservationsTrain = numel(sequencesTrain);
sequenceLengths = zeros(1,numObservationsTrain);

for i = 1:numObservationsTrain
    sequence = sequencesTrain{i};
    sequenceLengths(i) = size(sequence,2);
end

figure
histogram(sequenceLengths)
title("Sequence Lengths")
xlabel("Sequence Length")
ylabel("Frequency")
%%
maxLength = 400;
idx = sequenceLengths > maxLength;
sequencesTrain(idx) = [];
labelsTrain(idx) = [];
%%
numFeatures = size(sequencesTrain{1},1);
numClasses = numel(categories(labelsTrain));

layers = [
    sequenceInputLayer(numFeatures,'Name','sequence')
    bilstmLayer(2000,'OutputMode','last','Name','bilstm')
    dropoutLayer(0.5,'Name','drop')
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];

%%
miniBatchSize = 16;
numObservations = numel(sequencesTrain);
numIterationsPerEpoch = floor(numObservations / miniBatchSize);

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',1e-4, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{sequencesValidation,labelsValidation}, ...
    'ValidationFrequency',numIterationsPerEpoch, ...
    'Plots','training-progress', ...
    'Verbose',false);
%%
[netLSTM2,info] = trainNetwork(sequencesTrain,labelsTrain,layers,options);
%%
YPred_edge = classify(netLSTM2,sequencesValidation,'MiniBatchSize',miniBatchSize);
YValidation_edge = labelsValidation;
accuracy = mean(YPred_edge == YValidation_edge);

confusionchart(YValidation_edge,YPred_edge);
%%
cnnLayers = layerGraph(netCNN2);
layerNames = ["input_1" "fc1000" "fc1000_softmax" "ClassificationLayer_fc1000"];
%imageinputlayer,dropoutlayer,fullyconnected,softmaxlayer,classificationlayer
cnnLayers = removeLayers(cnnLayers,layerNames);
inputSize = netCNN2.Layers(1).InputSize(1:2);
averageImage = netCNN2.Layers(1).Mean;

inputLayer = sequenceInputLayer([inputSize 3], ...
    'Normalization','zerocenter', ...
    'Mean',averageImage, ...
    'Name','input');
layers = [
    inputLayer
    sequenceFoldingLayer('Name','fold')];

lgraph = addLayers(cnnLayers,layers);
lgraph = connectLayers(lgraph,"fold/out","conv1");
lstmLayers = netLSTM2.Layers;
lstmLayers(1) = [];

layers = [
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    lstmLayers];

lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,"avg_pool","unfold/in");

lgraph = connectLayers(lgraph,"fold/miniBatchSize","unfold/miniBatchSize");
net_rotate2 = assembleNetwork(lgraph);
%%
exportNetworkToTensorFlow(net_rotate2,"RotationError_model2")
exportNetworkToTensorFlow(netLSTM2,"RotationErrorTest_model")
%%
exportONNXNetwork(net_rotate,"rotation_error_net.onnx")
%%
filename = "under.mp4";
video = readVideo(filename);
video = imresize(video,inputSize);
[YPred,score] = classify(net_rotate2,{video});
YPred
score
%%
function [files, labels] = TrainingDataFiles(dataFolder)
fileExtension = ".mp4";
listing = dir(fullfile(dataFolder, "*", "*" + fileExtension));
numObservations = numel(listing);
files = strings(numObservations,1);
labels = cell(numObservations,1);
for i = 1:numObservations
    name = listing(i).name;
    folder = listing(i).folder;
    [~,labels{i}] = fileparts(folder);
    files(i) = fullfile(folder,name);
end
labels = categorical(labels);
end

function video = readVideo(filename)

vr = VideoReader(filename);
H = vr.Height;
W = vr.Width;
C = 3;

% Preallocate video array
numFrames = floor(vr.Duration * vr.FrameRate);
video = zeros(H,W,C,numFrames,'uint8');

% Read frames
i = 0;
while hasFrame(vr)
    i = i + 1;
    video(:,:,:,i) = readFrame(vr);
end

% Remove unallocated frames
if size(video,4) > i
    video(:,:,:,i+1:end) = [];
end

end
function videoResized = centerCrop(video,inputSize)

sz = size(video);

if sz(1) < sz(2)
    % Video is landscape
    idx = floor((sz(2) - sz(1))/2);
    video(:,1:(idx-1),:,:) = [];
    video(:,(sz(1)+1):end,:,:) = [];
    
elseif sz(2) < sz(1)
    % Video is portrait
    idx = floor((sz(1) - sz(2))/2);
    video(1:(idx-1),:,:,:) = [];
    video((sz(2)+1):end,:,:,:) = [];
end

videoResized = imresize(video,inputSize(1:2));

end
