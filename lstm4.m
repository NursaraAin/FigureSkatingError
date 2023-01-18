netCNN = googlenet;
% dataFolder=fullfile("C:\Users\nursa\OneDrive - Universiti Malaya\FigureSkatingError\");
path2=fullfile("C:\Users\nursa\OneDrive - Universiti Malaya\FigureSkatingError\under-rotation\");
[files,labels] = TrainingDataFiles(path2);

%%
% idx = 1;
% filename = files(idx);
% video = readVideo(filename);
% size(video)
%%
inputSize = netCNN.Layers(1).InputSize(1:2);
layerName = "pool5-7x7_s1";%globalavg2dpoolinglayer

tempFile = fullfile(path2);

overwriteSequences = true;

if exist(tempFile,'file') && ~overwriteSequences
    load(tempFile)
else
    numFiles = numel(files);
    sequences2 = cell(numFiles,1);
    
    for i = 1:numFiles
        fprintf("Reading file %d of %d...\n", i, numFiles)
        
        video = readVideo(files(i));
        video = imresize(video,inputSize);
        sequences2{i,1} = activations(netCNN,video,layerName,'OutputAs','columns');
        
    end
    % Save the sequences and the labels associated with them.
    save(tempFile,"sequences2","labels","-v7.3");
end
%sequences(1:10)
%%
numObservations = numel(sequences2);
idx = randperm(numObservations);
N = floor(0.8 * numObservations);

idxTrain = idx(1:N);
sequencesTrain2 = sequences2(idxTrain);
labelsTrain2 = labels(idxTrain);

idxValidation = idx(N+1:end);
sequencesValidation2 = sequences2(idxValidation);
labelsValidation2 = labels(idxValidation);
%%
numObservationsTrain = numel(sequencesTrain2);
sequenceLengths = zeros(1,numObservationsTrain);

for i = 1:numObservationsTrain
    sequence = sequencesTrain2{i};
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
sequencesTrain2(idx) = [];
labelsTrain2(idx) = [];
%%
numFeatures = size(sequencesTrain2{1},1);
numClasses = numel(categories(labelsTrain2));

layers = [
    sequenceInputLayer(numFeatures,'Name','sequence')
    bilstmLayer(2000,'OutputMode','last','Name','bilstm')
    dropoutLayer(0.5,'Name','drop')
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];

%%
miniBatchSize = 16;
numObservations = numel(sequencesTrain2);
numIterationsPerEpoch = floor(numObservations / miniBatchSize);

options = trainingOptions("adam", ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',1e-4, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{sequencesValidation2,labelsValidation2}, ...
    'ValidationFrequency',numIterationsPerEpoch, ...
    'Plots','training-progress', ...
    'Verbose',false);
%%
[netLSTM2,info] = trainNetwork(sequencesTrain2,labelsTrain2,layers,options);
%%
YPred_rotate = classify(netLSTM2,sequencesValidation2,'MiniBatchSize',miniBatchSize);
YValidation_rotate = labelsValidation2;
accuracy2 = mean(YPred_rotate == YValidation_rotate);
confusionchart(YValidation_rotate,YPred_rotate);
%%
cnnLayers = layerGraph(netCNN);
layerNames = ["data" "pool5-drop_7x7_s1" "loss3-classifier" "prob" "output"];
%imageinputlayer,dropoutlayer,fullyconnected,softmaxlayer,classificationlayer
cnnLayers = removeLayers(cnnLayers,layerNames);
inputSize = netCNN.Layers(1).InputSize(1:2);
averageImage = netCNN.Layers(1).Mean;

inputLayer = sequenceInputLayer([inputSize 3], ...
    'Normalization','zerocenter', ...
    'Mean',averageImage, ...
    'Name','input');
layers = [
    inputLayer
    sequenceFoldingLayer('Name','fold')];

lgraph = addLayers(cnnLayers,layers);
lgraph = connectLayers(lgraph,"fold/out","conv1-7x7_s2");%conv2dlayer
lstmLayers = netLSTM2.Layers;
lstmLayers(1) = [];

layers = [
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    lstmLayers];

lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,"pool5-7x7_s1","unfold/in");

lgraph = connectLayers(lgraph,"fold/miniBatchSize","unfold/miniBatchSize");
net_rotate = assembleNetwork(lgraph);
%%
exportNetworkToTensorFlow(net_rotate,"RotationError_model")
%exportNetworkToTensorFlow(netLSTM2,"RotationErrorTest_model")
%exportONNXNetwork(net_rotate,"rotation_error_net.onnx")
%%
filename = "under.mp4";
video = readVideo(filename);
video = centerCrop(video,inputSize);
[YPred,score] = classify(net_rotate,{video});
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
function [X] = preprocessUnlabeledVideos(XCell)
    % Pad the sequences with zeros in the fourth dimension (time) and
    % concatenate along the fifth dimension (batch)
    X = padsequences(XCell,4);
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