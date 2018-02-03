# TripletLoss
CNN Triplet Loss function implementation for matlab 

This is a matlab implementation of CNN (convolutional neural network)  triplet loss function, based 
 on the article "FaceNet: A Unified Embedding for Face Recognition
 and Clustering"  Google Inc 2015.
	
 The goal of this code is to use Pre-trained net for classification of new objects, that 
 does not include in the training set, without  training the net again. The classification is done
  based on extractining features from the net, then use  nearest neighbor distance to detrmine whether
  two objects are from the same or different classes. 
   for more information see the article  above.
	
 This is only preliminary, and could be improved a lot.
I would like to get any suggestions or help for optimize and improving the code. see the code on Git:
 GitHub: https://github.com/roytalman/TripletLoss.git

	
	Thanks
	Roy Talman
	roytalman@gmail.com
	Copyright (c) Roy Talman 2018
