Author: Prakhar Dogra
G Number: G01009586
CS 657 Assignment 3

The following README gives details about the files contained in this folder:

1. Dataset
	The dataset was downloaded from the Kaggle website from the following link : https://www.kaggle.com/c/nyc-taxi-trip-duration/data

2. SourceCode
	This folder contains the Source Code of all variations of Gradient Descent used to complete the tasks of the assignment.
	Use regresion to train a model that can predict the duration:
		- Using a version of regression that optimizes the parameters with Gradient Descent: GradientDescent.py
		- Using a version of regression that optimizes the parameters with Stochastic Gradient Descent: StochasticGradientDescent.py
		- Add L1 regularization to the model: GradientDescent_L1_Regression.py, StochasticGradientDescent_L1_Regression.py
		- Add L2 regularization to the model: GradientDescent_L2_Regression.py, StochasticGradientDescent_L2_Regression.py

3. PseudoCode
	This folder contains the Pseudo Codes for Feature Engineering, Cross Validation and Final Testing mentioned above:
	Following are the Pseudo Codes associated with their respective tasks:
	CrossValidation.pdf
		- Perform Cross Validation on 80% of dataset and find best parameters for all the models monetioned above.
	Training_and_Testing.pdf
		- Perform Training on 80% dataset followed by testing on remaining 20% of dataset using the best parameters for all the models monetioned above.

4. Graph
	This folder contains the graphs that were plotted for each respective task:
	Legend for all three graphs:
	BGD: Gradient Descent
	SGD: Stochastic Gradient Descent
	BGD_L1: Gradient Descent with L1 Regression
	BGD_L2: Gradient Descent with L2 Regression
	SGD_L1: Stochastic Gradient Descent with L1 Regression
	SGD_L2: Stochastic Gradient Descent with L2 Regression
	Graph #1
		- Bar graph to compare Training Times for all the 6 models mentioned above
	Graph #2
		- Bar graph to compare Root Mean Square Error for all the 6 models mentioned above
	Graph #3
		- Line plot to compare Running Times (for different dataset sizes) for all the 6 models mentioned above
	Each graph plotted is the time taken by the job versus the amount of the data set.
	The graph was plotted for 10%, 20%, 30%,....., 100% data sizes. (In the incements of 10%)
	
6. General Information
	- For each program, the size (in percent) of the dataset to be used is entered as command line arguement when executing the python file.
	- Initially all the programs were executed to perform Cross Validation and then the code to calculate RMSE and MAE for the test set was added at the end.