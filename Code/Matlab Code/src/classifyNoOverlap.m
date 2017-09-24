%%% This script is designed to classify images using the eigenfaces
%%% algorithm to compute a new facial recognition basis for all incoming
%%% images. An image that needs to be classified is projected into this new
%%% basis, with a configurable number of basis vectors, and then classified
%%% with an SVM. The SVM here is bases on the LIBCSVM library availible at
%%% https://www.csie.ntu.edu.tw/~cjlin/libsvm/
%%% 
%%%
%%% Requires that the included data file 'yaleData.mat' in the following
%%% relative directory from the working directory of this code:
%%% '../Data/yaleData.mat'
%%%
%%%
%%% This code is based off of some starter code written by Jaime I.
%%% Cervantes at https://github.com/JaimeIvanCervantes/FaceRecognition
%%%
%%%
%%% Author: Nicholas Renninger
%%% Last Updated: May 4, 2017
%%% Date Created: Aprl 17, 2017


% Housekeeping
close all
clear
clc

%% Setup and User Input
MARKERSIZE = 9;
MAX_MULTICLASS_COMPARE_ITER = 1000;
FONTSIZE = 24;
set(0, 'defaulttextinterpreter', 'latex');
saveLocation = '../../../Figures/Face_Matching/';
shouldSaveFigures = true;

% select how many principle eigenvectors  of the eigenface basis to use for
% classfication
NUM_EIG_FEATURES = 15;

   
disp('Loading database please wait...')

% loading in data set
load '../Data/yaleData.mat'
faceData = yaleData;

disp('Database active.') 

% number of classes and number of images per
numClasses = 68;
numImagesPerClass = 13;

% get user input
faceToMatch = input(['Enter the face # in the database you want to find', ...
                     ' matches for: ']);
                 


%% Eigenfaces Algorithm (SVD -> PCA)

% Read images in T matrix
[nRow, nCol, totalNumImages] = size(faceData);

% T is a matrix containing the reshaped vectors for each image
faceMatrix = reshape(faceData, [nRow*nCol, totalNumImages]);

% remove selected face from the training data to make sure the SVM is not
% classifying an exact match of the thing its trying to classify...
% confusing huh. Also remove one other image from each other class
% corresponding to the selected image's class index. i.e. if you selected
% the 15th image, delete every 2nd image from each class in the image
% database
indicesToDelete = mod(faceToMatch, numImagesPerClass):numImagesPerClass: ...
                      numClasses * numImagesPerClass;
indicesToDelete = indicesToDelete( indicesToDelete > 0 );
% A(:, indicesToDelete) = [];
faceMatrix(:, indicesToDelete) = [];

% update how many images there are, just deleted on image from each class
totalNumImages = totalNumImages - numClasses;
numImagesPerClass = numImagesPerClass - 1;

% phi is the mean of the entire set of training images
phi = mean(faceMatrix, 2);

% make a matrix with M colums, with each column being phi to subtract off
% the average features of each
psi = repmat(phi, 1, totalNumImages);

% substract mean to get a matrix of the distinguishing features (each row)
% of each face (each face is a col vec of A)
A = faceMatrix - psi;

% Reshape the selected face
selectedFace = reshape(faceData(:, :, faceToMatch), [nRow*nCol 1]);
distinguishingFeatures = selectedFace - phi;


% calculate the SVD matrix C = A'*A, which is the transpose of the
% covariance. Use A' * A to save a ton of computation time, as the
% eigenvectors of  A' * A are the same as the much larger matrix A * A'
C = A'*A;

% Obtaining eigenvalues and eigenvectors of C = A'*A
[eigVecs, eigValMat] = eig(C);

% Obtaning more relevant eigenvalues and eigenvectors
eigVals = diag(eigValMat);

principle_evals = [];
principle_evecs = [];

% perform PCA by ordering the eig vals and vecs by their importance
for i = totalNumImages:-1:numClasses + 1
    principle_evals = [principle_evals, eigVals(i)];
    principle_evecs = [principle_evecs, eigVecs(:,i)];
end

% Obtaining the eigenvectors
U = A * principle_evecs; 

% Obtaining PCA weights, multiply each eigenvector of U: u_i by the vector
% containing the distinguishing features of each input image: phi_i
Wpca = U' * A;
 

%% Classification with SVM 

% Obtain the weights of the normalized selected face
selectedFaceWeights = U' * distinguishingFeatures;

% Setting the SVM parameters. Uses an rbf kernel
K = 1e9;
gamma = 1/NUM_EIG_FEATURES;
inputParams = [' -t ' int2str(2) ' -c ' int2str(K) ' -b ' int2str(1) ...
               ' -g ' int2str(gamma)];

% prevArray starts with an array containing each class. Winner array is the
% classes that are selected in the binary tree
prevArr = 1:numClasses; 
winnerArr = [];

% This section of the code computes a binary SVM tree, by solving a 2 class
% problem with SVM for every 2 classes (1 and 2, 3 and 4, 5 and 6 etc). The
% that you selected is classified according to each 2-class problem and the
% class selected goes on to compete with the other classes selected.

currIter = 1;
CLASS = NaN;

while currIter < MAX_MULTICLASS_COMPARE_ITER
    
    winnerArr = []; 

    for winRep = 1:2:length(prevArr) 

        % Selects the two classes to train the SVM
        if winRep >= length(prevArr)
            i = prevArr(winRep) ;       
            j = prevArr(winRep - 1) ;   
        else
            i = prevArr(winRep)  ;      
            j = prevArr(winRep + 1)  ; 
        end

        % Selects the features of the 2 classes
        feature = [Wpca(1:NUM_EIG_FEATURES, ...
                        numImagesPerClass*i-(numImagesPerClass-1): ...
                        numImagesPerClass*i), ...
                   Wpca(1:NUM_EIG_FEATURES, ...
                        numImagesPerClass*j-(numImagesPerClass-1): ...
                        numImagesPerClass*j)]';
                    
        [numObservations, numFeatures] = size(feature);


        % scale each feature vector to a value between 0 and 1 
        for k = 1:numObservations
            features_normed(k, :) = (feature(k, :) - min(feature(k, :))) / ...
                                    ( max(feature(k, :)) - min(feature(k, :)) );
        end
          
        
        % Assigns the labels for each class
        for m1 = 1:numImagesPerClass
           label(m1) = 1; 
        end

        for n1 = numImagesPerClass+1:2*numImagesPerClass
           label(n1) = -1; 
        end
       
        
        % The SVM is trained
        model = svmtrain(label', features_normed, inputParams);

        % The face that the user selected is classified to any of the two
        % classes
        guessLab(1) = 1;
        featuresToLookFor = selectedFaceWeights(1:NUM_EIG_FEATURES)';

        % scale each feature vector to a value between 0 and 1
        featuresToLookFor = (featuresToLookFor - min(featuresToLookFor)) / ...
                            (( max(featuresToLookFor)) - min(featuresToLookFor));

        predLabel = svmpredict(guessLab', featuresToLookFor, model);
        
        % plot SVs
        %{
        figure
        hold on
        for i = 1:2
            feature(1:13, i) = (feature(1:13, i)) / ...
                                max(feature(1:13, i));
                         
            feature(13:26, i) = (feature(13:26, i)) / ...
                             ( max(feature(13:26, i)));
            
        end
        
        plot(feature(1:13, 1), feature(1:13, 2), 'bo', 'markersize', MARKERSIZE)
        plot(feature(13:26, 1), feature(13:26, 2), 'rx', 'markersize', MARKERSIZE)
        plot(model.SVs(:, 1), model.SVs(:, 2), 'kp', 'markersize', MARKERSIZE)
        title(sprintf('Class 1: %d, Class 2: %d', i, j))
        xlim([-6, 6])
        ylim([-4, 4])
        hold off
        %}
        
        % A winner class is selected
        if predLabel == 1 
            winnerArr = [winnerArr i];
        elseif predLabel == -1 
            winnerArr = [winnerArr j];
        end

        if winnerArr > 1
            for c1 = 2:length(winnerArr)
                if winnerArr(c1) == winnerArr(c1 - 1)
                   winnerArr(c1) = []; 
                end
            end
        end

    end

    prevArr = winnerArr;

   % if the multi class algorithm has selected only one winner return it 
   if length(winnerArr) < 2
       % This is the class that was selected
       CLASS = winnerArr;
       break
   end
   
   currIter = currIter + 1;
end
 
% check if the multi-class algorithm ever actually selected a face
if isnan(CLASS)
    error('The SVM never returned an absolute classification.')
end

disp('The class that matches your face is:')
disp(CLASS)

%% Plot the selected face
titleString = sprintf('Selected Face to Match - Image Number %d - Matching Class %d', ...
                      faceToMatch, CLASS);
saveTitle = cat(2, saveLocation, sprintf('%s.pdf', titleString));
                  
figure('name', titleString)

imagesc(reshape(selectedFace, nRow, nCol));
colormap gray;
title(titleString)
xlabel('Pixels - Horizontal')
ylabel('Pixels - Vertical')
set(gca, 'FontSize', round(FONTSIZE * 0.5))
set(gca, 'defaulttextinterpreter', 'latex')
set(gca, 'TickLabelInterpreter', 'latex')
axis equal

% setup and save figure as .pdf
saveMeSomeFigs(shouldSaveFigures, saveTitle)


%% setup plot saving
titleString = sprintf('Faces Matching Class %d', CLASS);
saveTitle = cat(2, saveLocation, sprintf('%s.pdf', titleString));

hFig = figure('name', titleString);
scrz = get(groot, 'ScreenSize');
set(hFig, 'Position', scrz)

numImgPRow = ceil(sqrt(numImagesPerClass));
if numImgPRow^2 == numImagesPerClass
    subPlotX = numImgPRow;
    subPlotY = numImgPRow;
elseif numImgPRow * (numImgPRow - 1) == numImagesPerClass
    subPlotX = numImgPRow;
    subPlotY = numImgPRow - 1;
end

for i = 1:numImagesPerClass
    
    subTitStr = sprintf('Face number %d', ...
                        winnerArr*(numImagesPerClass + 1) - i + 1);


    subplot(subPlotX, subPlotY, i)
    imagesc(reshape(faceMatrix(:, ...
                               winnerArr*numImagesPerClass - i + 1), ...
                    nRow, nCol));
    colormap gray;
    
    title(subTitStr)
    xlabel('Pixels - Horizontal')
    ylabel('Pixels - Vertical')
    axis equal
    set(gca, 'FontSize', round(FONTSIZE * 0.5))
    set(gca, 'defaulttextinterpreter', 'latex')
    set(gca, 'TickLabelInterpreter', 'latex')

end

h = gca;
leg = h.Legend;
titleStruct = h.Title;
set(titleStruct, 'FontWeight', 'bold')
set(leg, 'FontSize', round(FONTSIZE * 0.8))
mtit(titleString, 'Fontsize', FONTSIZE, 'FontWeight', 'bold');

% setup and save figure as .pdf
saveMeSomeFigs(shouldSaveFigures, saveTitle)

