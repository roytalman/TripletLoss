classdef TriplteLossLayer < nnet.layer.RegressionLayer
      
    
    % Define Triplet loss layer for classification 
    % Written by roy Talman 2/1/2018 , 
    % Email adress: roytalman@gmail.com
    % GitHub: https://github.com/roytalman/TripletLoss.git
    % Based on the article "FaceNet: A Unified Embedding for Face Recognition
    % and Clustering"  Google Inc 2015
    properties
        % (Optional) Layer properties

        % Layer properties go here
    end
 
    methods
        function layer = TriplteLossLayer(Name)           
            % (Optional) Create a myClassificationLayer
            % Set layer name
            if nargin == 1
                layer.Name = name;
            end

            % Set layer description
            layer.Description = 'Triplte loss layer';
            % Layer constructor function goes here
        end

        function loss = forwardLoss(layer, Y, T)
            % Return the loss between the predictions Y and the 
            % training targets T
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         loss  - Loss between Y and T
            
            % Sort triplte to anchor  ,Positive and Negative
            Anchor = Y(:,:,:,1:3:end) ;
            Pos = Y(:,:,:,2:3:end) ;
            Neg = Y(:,:,:,3:3:end) ;
            
            % Normlize to obtain Norm of 1
            Anchor_Norm = bsxfun(@rdivide ,Anchor,sqrt(sum(Anchor.^2))) ;
            Pos_Norm = bsxfun(@rdivide ,Pos,sqrt(sum(Pos.^2))) ;
            Neg_Norm = bsxfun(@rdivide ,Neg,sqrt(sum(Neg.^2))) ;            
            % Calculate Positive and negative distance
            PosDiff = sqrt(sum(squeeze(Anchor_Norm-Pos_Norm).^2));
            NegDiff = sqrt(sum(squeeze(Anchor_Norm-Neg_Norm).^2));
            % Layer forward loss function goes here
            loss = 0.2 -   median( NegDiff )+( median( PosDiff )  ) ;
%              disp(['pos dist: ' num2str(median( PosDiff  )) ' ,  Neg dist: ' num2str(median(NegDiff)) ' ,  Diff: ' num2str(median(NegDiff)-median(PosDiff))])
            loss = max(0,loss);
        end
        
        function dLdX = backwardLoss(layer, Y, T)
            % Backward propagate the derivative of the loss function
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         dLdX  - Derivative of the loss with respect to the input X        
            N = size(Y,4);
             % Sort triplte to anchor  ,Positive and Negative
            Anchor = Y(:,:,:,1:3:end) ;
            Pos = Y(:,:,:,2:3:end) ;
            Neg = Y(:,:,:,3:3:end) ;
            
            
            % Chack if positive distanse bigger then negative to set
            % grdient decsent direction
            DiffTriplte1 =  sign( Neg - Pos );
            DiffTriplte2 =  sign( Pos - Anchor );
            DiffTriplte3 =  sign( Neg - Anchor );
            
            DiffTriplte = gpuArray(zeros(size(DiffTriplte1).*([1 1 1 3])));
             % Duplicate three times direction
            DiffTriplte(:,:,:,1:size(DiffTriplte1,4)) = DiffTriplte1 ;
            DiffTriplte(:,:,:,(1:size(DiffTriplte1,4)) + size(DiffTriplte1,4)) = DiffTriplte2 ;
            DiffTriplte(:,:,:,(1:size(DiffTriplte1,4)) + 2*size(DiffTriplte1,4)) = DiffTriplte3 ;
            
            % Set Gradient
            dLdX = DiffTriplte/(N^2);
            % Layer backward loss function goes here
        end
    end
end