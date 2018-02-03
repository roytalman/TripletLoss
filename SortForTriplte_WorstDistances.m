function [ TriplteNewOrder ] = SortForTriplte_WorstDistances( PredictionTrain , TrainLabel)
     % Sort triplet loss data, after initial traning for re-training.
     % the data is sorted into triplets: anchor, Positive and negative.
     % The seperation done based on Positive cupple maximum distance and negative cupple minimum distance
     %  i.e the most difficalt task. Matched labels is sorted aswell.
     
     % input:  PredictionTrain - array of traning pictures
     %         TrainLabel - Pictures labels
     % Outputs: TriplteNewOrder - Index of new triplte data
     
NumSamp = size(PredictionTrain,1) ; 
DistMat = zeros( NumSamp , NumSamp , size(PredictionTrain,2));
for k = 1:size(PredictionTrain,2)
    DistMat(:,:,k)  = PredictionTrain(:,k)-PredictionTrain(:,k)' ;
end
Dist = sum(DistMat.^2,3) ;
NumSubSamp = 1000 ; 
 for k = 1:NumSamp
     User                   = TrainLabel(k);
     SameUserRec            = find(TrainLabel(k) == TrainLabel ) ;
     OtherRecs              = setdiff(  1:NumSamp,SameUserRec) ;
     OtherRecsSamp          = OtherRecs( randperm( length(OtherRecs) , NumSubSamp )) ;
     [val Pic1Ind ]         = max(Dist(k,SameUserRec)) ;
     [val Pic2Ind ]         = min(Dist(k,OtherRecsSamp)) ;
     TriplteNewOrder(3*k-2) = k ;
     TriplteNewOrder(3*k-1) = SameUserRec(Pic1Ind) ;
     TriplteNewOrder(3*k)   = OtherRecs(Pic2Ind) ;

 end