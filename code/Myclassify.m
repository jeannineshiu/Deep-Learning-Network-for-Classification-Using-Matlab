function [predictedLabel,filenames] = Myclassify(foldername)

load('trained_net3.mat');
%net name: CNNnet3

digitDatasetPath = fullfile(foldername);

testData = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

labelCount = countEachLabel(testData);

labels = classify(CNNnet3,testData);

filenames = testData.Files;

for ii = 1:labelCount.Count
   if char(labels(ii))=="NG"
       predictedLabel(ii) = 0;
   else
       predictedLabel(ii) = 1;
   end
end

predictedLabel = predictedLabel';

end