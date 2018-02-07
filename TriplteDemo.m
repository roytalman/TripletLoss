
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
    Pics((1:5)+5*k,(1:5)+5*k, (1:NumPicPerGroups)+ NumPicPerGroups*(k-1) ) = ...
    Pics((1:5)+5*k,(1:5)+5*k, (1:NumPicPerGroups)+ NumPicPerGroups*(k-1) ) + 3 ;
    Labels((1:NumPicPerGroups)+ NumPicPerGroups*(k-1)) = k ; % image group label
end

% train and test set:
NumTrain = 0.7 ; % 70% data for traning
% [val RandomIndex ] = sort(rand(1,TotalPic)) ;  % random index
TrainPic           = Pics(:,:,(1:NumTrain*TotalPic)) ; % 70% training set, Classes 1-7 
TrainLabel         = Labels((1:NumTrain*TotalPic)) ;
TestPic            = Pics(:,:,( NumTrain*TotalPic + 1 : end )) ; % 30% testing set , Classes 8-10
TestLabels         = Labels(( NumTrain*TotalPic + 1 : end )) ;

% plot train images
figure
for k =1:9
    subplot(3,3,k)
    RandInd = randperm(size(TrainPic,3),1) ;
    imagesc(TrainPic(:,:,RandInd)) ;
    title([ 'Train Image ' num2str( RandInd) ]) 
end
% sort in triplte : 
[ PicsTrainTriplet , LabelsTrainTriplte ] = SortForTriplte( TrainPic , TrainLabel ) ;
[ PicsTestTriplet , LabelsTestTriplte ]   = SortForTriplte( TestPic , TestLabels ) ; 


% define CNN pathology

 NumFeatures = 16; % featers for catgory saparation, originaly 128 
 TriplteLayers = GetLayers(ImageSize,NumFeatures) ; % Net pathology

 options = trainingOptions('sgdm','MaxEpochs',15, 'InitialLearnRate',0.001,'Momentum',0.99...
    ,'MiniBatchSize',300,'Shuffle','never','Plots','training-progress','ExecutionEnvironment','gpu');  % CNN trainig option. Vary improtant-  "Shuffle" shouled be set to 'never'!

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
options = trainingOptions('sgdm','MaxEpochs',60, 'InitialLearnRate',0.001...
    ,'MiniBatchSize',300,'Shuffle','never','Plots','training-progress','ExecutionEnvironment','gpu');  % CNN trainig option Vary improtant-  "Shuffle" shouled be set to never!

LabelsForNet = zeros(size(RetrainPictures,4),NumFeatures ) ; % this is not needed in triplter since we calculating loss using triplte traning set

%%% training %%%
Net_Retrain = trainNetwork( PicsTrainTriplet , LabelsForNet, Net.Layers  ,options) ;
[SumGood_2,NegDist,PosDist] = TripletResultAnalyze(Net_Retrain,PicsTestTriplet);

figure
for k = 1:3 
    RandInd = randperm(size(PicsTestTriplet,4)/3,1) ;
    subplot(3,3,k*3-2); imagesc( PicsTestTriplet(:,:,1,3*RandInd-2) );title(['Anchor Pic, triplet ' num2str(RandInd)])
    subplot(3,3,k*3-1); imagesc( PicsTestTriplet(:,:,1,3*RandInd-1) );title(['Pos Pic, Dist from Anchor: ' num2str(PosDist(RandInd)) ])
    subplot(3,3,k*3); imagesc( PicsTestTriplet(:,:,1,3*RandInd) );title(['Neg Pic, Dist from Anchor: ' num2str(NegDist(RandInd)) ])
end

