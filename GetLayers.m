function  [Layers] = GetLayers(ImageSize , NumFeatures)
% define CNN pathology
 Layers = [imageInputLayer([ImageSize ImageSize 1]);
       convolution2dLayer([5 5],256,'Padding',[0 0]);
       reluLayer();
       maxPooling2dLayer([3 3],'Stride',[3 3],'Padding',[0 0]);
%        convolution2dLayer([3 3],256,'Padding',[0 0]);
%        reluLayer();
        dropoutLayer(0.4)
        fullyConnectedLayer(NumFeatures);
       TriplteLossLayer];
end