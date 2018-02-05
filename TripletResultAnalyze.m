function [SumGood,NegDist,PosDist] = TripletResultAnalyze(Net,PicsTestTriplet)

% prediction for test triplets:
Prediction      = predict(Net,PicsTestTriplet);
NegDist         = sum((Prediction(1:3:end,:)-Prediction(3:3:end,:)).^2,2) ; 
PosDist         = sum((Prediction(1:3:end,:)-Prediction(2:3:end,:)).^2,2)  ;
Prediction_diff = NegDist -  PosDist ;

% Negative distance minus Positive distanse histogram (should be greater then 0)
 figure;hist( Prediction_diff,80) ; title('Diff Distance histogram (Negative - Positive, random triplets)')
 xlabel('Distances difference (Negative - Positive)')
 ylabel('Count')

 % Precntage of distances greater then 0
SumGood = sum(Prediction_diff>0)./length(Prediction_diff);

end