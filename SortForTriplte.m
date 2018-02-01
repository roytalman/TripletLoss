 function [ PicsOut , LabelsOut ] = SortForTriplte(Pics , Labels)
     % sort triplet loss data , the data is sorted into triplets: anchor,
     % Positive and negative. Matched labels is sorted aswell
    PicsOut = zeros(size(Pics,1),size(Pics,1),1,size(Pics,3));
     for k = 1:size(Pics,3)
         User                 = Labels(k);
         SameUserRec          = find(Labels(k) == Labels ) ;
         OtherRecs            = setdiff(  1:size(Pics,3),SameUserRec) ;
         Pic1Ind              = randperm(length(SameUserRec),1 );
         Pic2Ind              = randperm(length(OtherRecs),1 );
         PicsOut(:,:,1,3*k-2) = Pics(:,:,k) ;
         PicsOut(:,:,1,3*k-1) = Pics(:,:,SameUserRec(Pic1Ind),:) ;
         PicsOut(:,:,1,3*k)   = Pics(:,:,OtherRecs(Pic2Ind),:) ;
         LabelsOut(3*k-(1:2)) = Labels(k);
         LabelsOut(3*k)       = Labels(OtherRecs(Pic2Ind));
     end


 end