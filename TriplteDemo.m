
%%%  triplte Demo %%%

% Written by Roy Talman 2/1/2018 , 
% Email adress: roytalman@gmail.com
% GitHub: https://github.com/roytalman/TripletLoss.git
% Based on the article "FaceNet: A Unified Embedding for Face Recognition
% and Clustering"  Google Inc 2015
% Copyright (c) Roy Talman 2018

% image settings:
ImageSize       = 64 ; 
NumGroups       = 10 ;
NumPicPerGroups = 1000 ;
TotalPic = NumGroups*NumPicPerGroups ;
% create random data, 10000 pictures with the size 64*64:
Pics = normrnd(0,1,[ImageSize ImageSize TotalPic]);

% add  feature for groups of 1000 piuctures, ten different types of pictures:
for k = 1:10 
    Pics((1:5)+3*k,(1:5)+3*k, (1:NumPicPerGroups)+ NumPicPerGroups*(k-1) ) = ...
        Pics((1:5)+3*k,(1:5)+3*k, (1:NumPicPerGroups)+ NumPicPerGroups*(k-1) ) +4 ;
    Labels((1:NumPicPerGroups)+ NumPicPerGroups*(k-1)) = k ; % image group label
end

% train and test set:
NumTrain = 0.7 ; % 70% data for traning
% [val RandomIndex ] = sort(rand(1,TotalPic)) ;  % random index
TrainPic           = Pics(:,:,(1:NumTrain*TotalPic)) ; % 70% training set, Classes 1-7 
TrainLabel         = Labels((1:NumTrain*TotalPic)) ;
TestPic            = Pics(:,:,( NumTrain*TotalPic + 1 : end )) ; % 30% testing set , Classes 8-10
TestLabels         = Labels(( NumTrain*TotalPic + 1 : end )) ;


% sort in triplte : 
[ PicsTrainTriplet , LabelsTrainTriplte ] = SortForTriplte( TrainPic , TrainLabel ) ;
[ PicsTestTriplet , LabelsTestTriplte ]   = SortForTriplte( TestPic , TestLabels ) ; 


% define CNN pathology

 NumFeatures = 16; % featers for saparation, originaly 128 
 TriplteLayers = [imageInputLayer([ImageSize ImageSize 1]);
       convolution2dLayer([5 5],32,'Padding',[0 0]);
       reluLayer();
       maxPooling2dLayer([3 3],'Stride',[3 3],'Padding',[0 0]);
            convolution2dLayer([3 3],32,'Padding',[0 0]);
       reluLayer();
        dropoutLayer(0.4)
        fullyConnectedLayer(NumFeatures);
       TriplteLossLayer]
   
 options = trainingOptions('sgdm','MaxEpochs',55, 'InitialLearnRate',0.001...
    ,'MiniBatchSize',300,'Shuffle','never','Plots','training-progress','ExecutionEnvironment','gpu');  % CNN trainig option Vary improtant-  "Shuffle" shouled be set to never!

LabelsForNet = zeros(size(PicsTrainTriplet,4),NumFeatures ) ; % this is not needed in triplter since we calculating loss using triplte traning set

%%% training %%%
Net = trainNetwork( PicsTrainTriplet , LabelsForNet, TriplteLayers  ,options) ;

% prediction for test triplets:
[SumGood] = TripletResultAnalyze(Net,PicsTestTriplet)

%%% Retrain for Worst training cases %%%
PredictionTrain     = predict(Net,permute(TrainPic,[1 2 4 3]));
[ TriplteNewOrder ] = SortForTriplte_WorstDistances(PredictionTrain , TrainLabel) ;
   
RetrainLabels    = TrainLabel(TriplteNewOrder) ;
RetrainPictures  = permute(TrainPic(:,:,RetrainLabels),[1 2 4 3]);
options = trainingOptions('sgdm','MaxEpochs',50, 'InitialLearnRate',0.001...
    ,'MiniBatchSize',300,'Shuffle','never','Plots','training-progress','ExecutionEnvironment','gpu');  % CNN trainig option Vary improtant-  "Shuffle" shouled be set to never!

LabelsForNet = zeros(size(RetrainPictures,4),NumFeatures ) ; % this is not needed in triplter since we calculating loss using triplte traning set

%%% training %%%
Net_Retrain = trainNetwork( PicsTrainTriplet , LabelsForNet, Net.Layers  ,options) ;
[SumGood_2] = TripletResultAnalyze(Net_Retrain,PicsTestTriplet)
