Introduction
============

This project implements the Eigenface Analysis (a PCA variant) and Support Vector Machines (SVMs) to classify chosen faces picked out of the training set. It is an extension of code written by https://github.com/JaimeIvanCervantes.

The classsification itself code was written in Matlab, and it requires the user to install [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/). A database of facial imagery is included with subjects in various poses and lighting conditions. This data has already been scaled to a normalized size (images of size N x N).

Theory
======

Eigenface Analysis
-------------------------------------

Eigenface Analysis, or Principle Component Analysis, in order to reduce dimensionality and extract relevant features from the given image training data set. It is assumed that the datasets contain K persons or classes, each with n images of size Nx x Ny.  M is the total number of images, and is equal to K*n. 

The first step is to represent the Nx x Ny matrices in the form of Ti = Nx Ny X 1 vectors.  The next step is to subtract the mean of the entire set of images to each face, and  then we can proceed to find the eigenvalues and eigenvectors, with the top k eigenvectors of the data set being the k eigenfaces.

A = [Φi … ΦM] and the covariance matrix C = A*A’.  In order to obtain the eigenvalues and eigenvectors of C, the eigenvectors vi of the alternative matrix A’*A are obtained, and the eigenvectors of C are given by ui = A*vi. What we have done is change the basis of the problem from basis vectors that do not correspond to any sort of meanigful information, to a set of vectors (eigenface vectors) that represent the "direction" of important facial features.  Now each face in the training set (minus the mean) can be represented as a linear combination of these eigenface eigenvectors, with the weights given by wi_pca = ui * Φi. Each normalized image is then represented as a collection of these weights W  = [wi … wk].

Support Vector Machines
-----------------------

After the PCA analysis is performed on the data, classification with support vector machines was tested. To get more information about this proccess, I recommend reading the documents in the Reference Papers/ directory.

Several strategies to perform multi-class classification with SVM exist. The common "one-against-all" method is one of them. A bottom up binary tree classification was used in this project in order to reduce the problem to a two class problem. 

Results
=======
* mainClassifierSVM will take a single photo from the database, remove it from the training set, train the SVM, and then predict which class the chosen photo belongs to.
![Figure 1](Figures/Faces Matching Class 1.pdf)

* mainDecisionBoundaryPlot will take a user-inputted number of classes to work with. For each class:
1. Create a logical vector (indx) indicating whether an observation is a member of the class.
2. Train an SVM classifier using the predictor data and indx.
3. Store the classifier in a cell of a cell array.
  * Then, create a mesh grid over the normalized 2-D feature space and calculate the score of each class at each point on the grid using the model for each class. The index of the element with the largest score is the index of the class to which the new class observation most likely belongs. Plot the results.


Running the Code
================

1. To run the code, run either mainClassifierSVM.m or mainDecisionBoundaryPlot.m. 

2. The data has to be in the form of a Rows*Columns*M matrix.

3. The code implements the library LIBSVM. It is included in Code\Matlab Code\libsvm-3_22\*
