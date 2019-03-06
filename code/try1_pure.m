digitDatasetPath = fullfile('D:','桌面用', ...
    '程語','專題','data','Joint_resize');

digitData = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

labelCount = countEachLabel(digitData)

img = readimage(digitData,1);
size(img)

%trainNumFiles = 6180;
[trainDigitData,valDigitData] = splitEachLabel(digitData,0.83);
%[imds60, imds10, imds30] = splitEachLabel(imds,0.6,0.1)

%trainNumFiles = 750;
%[trainDigitData,valDigitData] = splitEachLabel(digitData,trainNumFiles,'randomize');

layers = [
	%The digit data consists of grayscale images, so the channel size (color channel) is 1. For a color image, the channel size is 3, corresponding to the RGB values.
    imageInputLayer([70 70 3])
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    %In this example, the output size is 2, corresponding to the 2 classes. 
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];


options = trainingOptions('sgdm',...
    'MaxEpochs',3, ...
    'ValidationData',valDigitData,...
    'ValidationFrequency',30,...
    'Verbose',false,...
    'Plots','training-progress');


net = trainNetwork(trainDigitData,layers,options);

CNNnet3 = net;
save('D:\桌面用\程語\專題\trained_net3.mat','CNNnet3');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% predictedLabels = classify(net,valDigitData);
% valLabels = valDigitData.Labels;

% accuracy = sum(predictedLabels == valLabels)/numel(valLabels)
