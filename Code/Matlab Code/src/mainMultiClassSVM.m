%% Find Multiple Class Boundaries Using Binary SVM
% AUTHOR: Nick Renninger, Richard Moon
% LAST UPDATED: April 28, 2017
% DESCRIPTION: This code implements PCA (eigenfaces), Fisherfaces (linear
% discriminant analysis) and an SVM to classify regions of the face space
% as characteristic of certain classes.

% Housekeeping
close all
clear
clc

%% Setup

%%%%%%%%%%%%%%%%%%%%%%%% Plot Setup %%%%%%%%%%%%%%%%%%%%%%%%
set(0, 'defaulttextinterpreter', 'latex');
saveLocation = '../../../Figures/';
LINEWIDTH = 2;
MARKERSIZE = 18;
FONTSIZE = 24;
colorVecsOrig = [0.294118 0 0.509804; % indigo
                 0.1 0.1 0.1; % orange red
                 1 0.843137 0; % gold
                 0.180392 0.545098 0.341176; % sea green
                 0.662745 0.662745 0.662745]; % dark grey 

% de-saturate background colors. saturation must be from 0-1
saturation = 0.5;
colorVecs = colorVecsOrig * saturation;
             
         
markers = {'o','*','s','.','x','d','^','+','v','>','<','p','h'};

shouldSaveFigures = true;

% Choose simple, pose variation, or illumination variation dataset 
numClasses = input(['Please select the number of classes to compare ', ...
                   '(recommended is 4 classes: 7, 22, 24, 46): \n']);

for i = 1:numClasses
    
    % enter the class number (ranges from 1-68 for pose.mat)
    whichClasses(i) = input('Input one class you want to compare: \n');
    
end

whichClasses = sort(whichClasses);
dataset = 2; % automatically choose pose.mat data set

if dataset == 1
    
     load '../Data/simple.mat'
     %Define variables
     numClasses = 200;% Number of classes
     numImagesPerClass = 3;% Number of images per class 
     
elseif dataset == 2
    
     load '../Data/yaleData.mat'
     face = yaleData;
     numImagesPerClass = 13;% Number of images per class
else
    disp('Input not valid');
end

testNum = input(['Please select whether to use Fischer Faces:', ...
                '\n (1) Use Eigenfaces only', ...
                '\n (2) Use Fischerfaces and Eigenfaces (more accurate)\n']);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Principal Component Analysis (PCA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Loading please wait...')

% Read images in T matrix
[nRow, nCol, maxNumImgPerClass, maxNumClasses] = size(face);

maxNumImages = maxNumImgPerClass * maxNumClasses;

totalNumImages = numClasses * numImagesPerClass;

% T is a matrix containing the reshaped vectors for each image
faceMatrix = reshape(face, [nRow*nCol, maxNumImages]);

% phi is the mean of the entire set of training images
phi = mean(faceMatrix, 2);

% make a matrix with maxNumImages colums, with each column being phi to
% subtract off the average features of each
psi = repmat(phi, 1, maxNumImages);

% substract mean
A = faceMatrix - psi;

% calculate the SVD matrix C = A'*A
C = A'*A;

% Obtaining eigenvalues and eigenvectors of C = A'A
[V, D] = eig(C);

% Obtaning more relevant eigenvalues and eigenvectors
eval = diag(D);

% initialize
p_eval = [];
p_evec = [];

% sorting eigenvalues from largest to smallest. Takes away zero eigenvalues
for i = maxNumImages:-1:maxNumClasses+1
    p_eval = [p_eval, eval(i)];
    p_evec = [p_evec, V(:,i)];
end

% Obtaining the eigenvectors of A*A'
U = A * p_evec; 

% Obtaining PCA weights, multiply each eigenvector of U: u_i by the vector
% containing the distinguishing features of each input image: phi_i
Wpca = U' * A;



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fisher's Linear Discriminant Analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
% Obtaining Sb and Sw
cMean = zeros(maxNumImages - maxNumClasses, maxNumImages - maxNumClasses);
Sb = zeros(maxNumImages - maxNumClasses, maxNumImages - maxNumClasses);
Sw = zeros(maxNumImages - maxNumClasses, maxNumImages - maxNumClasses);

pcaMean = mean(Wpca, 2);

for i = 1:maxNumClasses
    cMean = mean(Wpca(:, numImagesPerClass*i - (numImagesPerClass-1) : ...
                         numImagesPerClass*i), 2);
    Sb = Sb + (cMean-pcaMean)*(cMean-pcaMean)';
end

Sb = numImagesPerClass*Sb;

for i = 1:maxNumClasses
    cMean = mean(Wpca(:, numImagesPerClass*i - (numImagesPerClass-1) : ...
                         numImagesPerClass*i), 2);
    for j = numImagesPerClass*i - (numImagesPerClass-1):numImagesPerClass*i
         Sw = Sw + (Wpca(:,j) - cMean) * (Wpca(:,j) - cMean)';
    end
end

% Obtaining Fisher eigenvectors and eigenvalues
[Vf, Df] = eig(Sb, Sw);

% Calculating weights
Df = fliplr(diag(Df));
Vf = fliplr(Vf);

% Calculating fisher weights
Wf = Vf' * Wpca;

%% Choose to use Fischer Faces or not

% pick out indicies of chosen class images to compare
imagesPerClassVec = 1:1:13;
class_idxs = [];

% get the indicies of the selected classes
for i = 1:length(whichClasses)

    class_idxs = [class_idxs, (whichClasses(i) - 1) * ...
                              numImagesPerClass + imagesPerClassVec];
end

% select eigen or fischer faces
if testNum == 1 % use only eigen faces
    
    % define the features of the selected classes using 2 eigenvectors (1:2)
    features = Wpca(1:2, class_idxs)';
    face_basis = 'Eigenfaces';
    
else % or use fischer faces as well
    
    % define the features of the selected classes using 2 eigenvectors (1:2)
    features = Wf(1:2, class_idxs)';
    face_basis = 'Fisherfaces';
end

[~, numFeatures] = size(features);

% scale each feature vector to a value between 0 and 1
for i = 1:numFeatures
    features_normed(:, i) = (features(:, i) - min(features(:, i))) / ...
                            ( max(features(:, i)) - min(features(:, i)) );
end


disp('Loaded. Extracting Features and Projecting to new basis...')
%% Extracting and Labeling Data

% extract two principle eigenvalue weight vectors from the PCA weight
% matrix to form two columns of data. First column corresponds to the
% principle weight of each image in the eigenface space. Second column
% corresponds to the 2nd principle weight of each image in the egenface
% space.


% initialize
classes = cell(1, maxNumClasses);
Y = cell(1, maxNumImages);

% create class names
for i = 1:maxNumClasses
    classes{i} = sprintf('B0b %d', i);
end

% label each image into the class it is
for i = 1:numImagesPerClass:maxNumImages
    
    % label numImagesPerClass Images as same class, then move on to the
    % next label
    classes_idx = (i + numImagesPerClass - 1) / numImagesPerClass;
    Y(i:i + numImagesPerClass - 1) = classes(classes_idx);
    
end

% only pull out the labels relevant to the selected classes
Y = Y(class_idxs)';

%% Scatter Plot of the data.
titleString = sprintf('Scatter Plot - %s - Classes', face_basis);
for i = 1:numClasses
    titleString = sprintf('%s %d', titleString, whichClasses(i));
end

% setup plot saving
saveTitle = cat(2, saveLocation, sprintf('%s.pdf', titleString));


hFig = figure('name', titleString);
scrz = get(groot, 'ScreenSize');
set(hFig, 'Position', scrz)

g = gscatter(features_normed(:,1), features_normed(:,2), Y, ...
         linspecer(numClasses));
     

h = gca;
leg = h.Legend;
titleStruct = h.Title;
set(titleStruct, 'FontWeight', 'bold')
set(gca, 'FontSize', FONTSIZE)
set(leg, 'FontSize', round(FONTSIZE * 0.7))
set(gca, 'defaulttextinterpreter', 'latex')
set(gca, 'TickLabelInterpreter', 'latex')
for i = 1:numClasses
    g(i).LineWidth = LINEWIDTH;
    g(i).MarkerSize = MARKERSIZE;
    g(i).Marker = markers{i};
end

titleString = sprintf('Scatter Plot (%s) - Classes:', face_basis);
for i = 1:numClasses
    if i == numClasses
        titleString = sprintf('%s %d', titleString, whichClasses(i));
    else
        titleString = sprintf('%s %d,', titleString, whichClasses(i));
    end
end

title(titleString);
xlabel('Eigenvector Basis Element 1');
ylabel('Eigenvector Basis Element 2');
set(leg, 'Location', 'best', 'Interpreter', 'latex')
xlim([0, 1])
ylim([0, 1])
grid on


% setup and save figure as .pdf
saveMeSomeFigs(shouldSaveFigures, saveTitle)

disp('Done. Plotting projected data and beginning classification...')


%% Build SVM Models

% For each class:
%
% # Create a logical vector (|indx|) indicating whether an observation is
% a member of the class.
% # Train an SVM classifier using the predictor data and |indx|.
% # Store the classifier in a cell of a cell array. 
%
% It is good practice to define the class order.
SVMModels = cell(numClasses, 1);
classes = unique(Y);
rng(1); % For reproducibility

for j = 1:numel(classes)
    % Create binary classes for each classifier
    indx = strcmp(Y, classes{j}); 
    
    % fit a model to each binary set of data and store the resultant model
    SVMModels{j} = fitcsvm(features_normed, indx, ...
                           'ClassNames', [false true], ...
                           'Standardize', true, ...
                           'KernelFunction', 'rbf', ...
                           'OptimizeHyperparameters', 'auto', ...
                           'BoxConstraint', 1);
end

%%
% |SVMModels| is a numClasses-by-1 cell array, with each cell containing a
% |ClassificationSVM| classifier.  For each cell, the positive class is
% defined by the Y vector.
%%
% Define a fine grid within the plot, and treat the coordinates as new
% observations from the distribution of the training data.  Estimate the
% score of the new observations using each classifier.
mesh_scale_factor = 1e5;
mesh_steps = totalNumImages / mesh_scale_factor;
[x1Grid, x2Grid] = meshgrid(min(features_normed(:,1)):mesh_steps: ...
                            max(features_normed(:,1)), ...
                            min(features_normed(:,2)):mesh_steps: ...
                            max(features_normed(:,2)));
xGrid = [x1Grid(:), x2Grid(:)];
N = size(xGrid, 1);
Scores = zeros(N, numel(classes));

for j = 1:numel(classes)
    [~, score] = predict(SVMModels{j}, xGrid);
    % Second column contains positive-class scores
    Scores(:, j) = score(:, 2); 
end

%%
% Each row of |Scores| contains numClasses scores.  The index of the
% element with the largest score is the index of the class to which the new
% class observation most likely belongs.
%%

% Associate each new observation with the classifier that gives it the
% maximum score.  
[~, maxScore] = max(Scores,[],2);

%%
% Color in the regions of the plot based on which class the
% corresponding new observation belongs.

titleString = sprintf('Classification Regions - %s - Classes', face_basis);
for i = 1:numClasses
    titleString = sprintf('%s %d', titleString, whichClasses(i));
end

% setup plot saving
saveTitle = cat(2, saveLocation, sprintf('%s.pdf', titleString));


hFig = figure('name', titleString);
scrz = get(groot, 'ScreenSize');
set(hFig, 'Position', scrz)

g(1:numClasses) = gscatter(xGrid(:,1), xGrid(:,2), maxScore,...
                           colorVecs(1:numClasses, :));
hold on
g(numClasses+1:2*numClasses) = gscatter(features_normed(:,1), ...
                                        features_normed(:,2), ...
                                        Y, linspecer(numClasses));
h = gca;
leg = h.Legend;
titleStruct = h.Title;
set(titleStruct, 'FontWeight', 'bold')
set(gca, 'FontSize', FONTSIZE)
set(leg, 'FontSize', round(FONTSIZE * 0.8))
set(gca, 'defaulttextinterpreter', 'latex')
set(gca, 'TickLabelInterpreter', 'latex')
for i = numClasses+1:2*numClasses
    g(i).LineWidth = LINEWIDTH;
    g(i).MarkerSize = MARKERSIZE;
    g(i).Marker = markers{i - numClasses};
end

titleString = sprintf('Classification Regions (%s) - Classes:', face_basis);
for i = 1:numClasses
    if i == numClasses
        titleString = sprintf('%s %d', titleString, whichClasses(i));
    else
        titleString = sprintf('%s %d,', titleString, whichClasses(i));
    end
end

title(titleString);
xlabel('Eigenvector Basis Element 1');
ylabel('Eigenvector Basis Element 2');
set(leg, 'Location', 'best', 'Interpreter', 'latex')
grid on
xlim([0, 1])
ylim([0, 1])
hold off

for i = 1:numClasses
%     leg_string{i} = sprintf('%s Region', classes{i});
    leg_string{i} = sprintf('Actual %s Observations', ...
                                       classes{i});
end

legend(g(numClasses+1:2*numClasses), leg_string, 'Location', 'best', 'interpreter', 'latex');


% setup and save figure as .pdf
saveMeSomeFigs(shouldSaveFigures, saveTitle)

disp('Classification Completed. Plotting Results and Exiting.')

%% Plot the selected classes

for i = 1:numClasses
    titleString = sprintf('Face Image - Class %d - B0b %d', ...
                          whichClasses(i), whichClasses(i));

    % setup plot saving
    saveTitle = cat(2, saveLocation, sprintf('%s.pdf', titleString));
    
    hFig = figure('name', titleString);
    scrz = get(groot, 'ScreenSize');
    set(hFig, 'Position', scrz)
    
    for j = 1:numImagesPerClass
        
        subplot(round(sqrt(numImagesPerClass)), ...
                round(sqrt(numImagesPerClass)), j)
        imagesc(reshape(faceMatrix(:, (whichClasses(i) - 1) * maxNumImgPerClass + j), ...
                        nRow, nCol));
        colormap gray;
        
        xlabel('Pixels - Horizontal')
        ylabel('Pixels - Vertical')
        axis equal
        set(gca, 'FontSize', round(FONTSIZE * 0.5))
        
    end
    h = gca;
    leg = h.Legend;
    titleStruct = h.Title;
    set(titleStruct, 'FontWeight', 'bold')
    set(leg, 'FontSize', round(FONTSIZE * 0.8))
    set(gca, 'defaulttextinterpreter', 'latex')
    set(gca, 'TickLabelInterpreter', 'latex')
    titleString = sprintf('Face Image - Class %d - ``B0b %d"', ...
                          whichClasses(i), whichClasses(i));
    mtit(titleString, 'Fontsize', FONTSIZE, 'FontWeight', 'bold');
   
    
    
    % setup and save figure as .pdf
    saveMeSomeFigs(shouldSaveFigures, saveTitle)
end