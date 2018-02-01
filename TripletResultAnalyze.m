function [SumGood] = TripletResultAnalyze(Net,PicsTestTriplet)

% prediction for test triplets:
Prediction = predict(Net,PicsTestTriplet);
Prediction_diff = sum((Prediction(1:3:end,:)-Prediction(3:3:end,:)).^2,2) - sum((Prediction(1:3:end,:)-Prediction(2:3:end,:)).^2,2) ;

% Negative distance minus Positive distanse histogram (should be greater then 0)
 hist( Prediction_diff,80) ; title('Diff Distance (Negative - Positive  , random triplets)')
 % Precntage of distances greater then 0
SumGood = sum(Prediction_diff>0)./length(Prediction_diff);