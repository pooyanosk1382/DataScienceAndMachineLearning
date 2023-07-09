# DataScienceAndMachineLearning
This repository contains codes and summary of each lesson of the machine learning course.


# Data preprocessing in python:


### Libraries:
There are some useful libraries such as numpy, matplotlib, pandas and the most useful **sklearn**.

### Functions:
__SimpleImputer__ is used to take care of missing data.

__OneHotEncoder__ changes strings to numerical vectors.

__ColumnTransformer__ changes columns of dataset.

__LabelEncoder__ is used to convert two-state data to 0 and 1.

__train_test_split__ will split the whole dataset to two main parts, **Train & Test**.

__StandardScaler__ is used to apply standardization on data.


# Regression


## Simple Linear Regression:

### Functions:
__LinearRegression__ is an object to fit a regression on a train set.

__.predict__ is a function to predict the y by use of regression object.

## Multiple Linear Regression
The functions are just like __Simple Linear Regression__ just need to encode the non-numerical column by OneHotEncoder.

__Tip1:__ The LinearRegression function handle dummy variable and choose best feature.

__Tip2:__ Because the number of elements are more than two we can't show them by graphs ,so we just check the prediction with test set.


## Polynomial Regression

For this part we do just as same the multiple linear regression just create x^2 and x^3 and ... by PolynomialFeatures.

### Functions:
__PolynomialFeatures__ is used to create x and x^2and x^3 and ...


## Support Vector Regression
The first thing to mention is that we use feature scaling on both x and y because this is how svr works.

### Functions:
__SVR__ that is a class ,and we should make an object from this.

__inverse_transform__ that is used to return the values to the state before feature scaling


## Decision Tree Regression
This model is not good for one feature and is better to use it when we have some features. This doesn't need feature scaling. Other things are same as last parts.

### Functions:
__DecisionTreeRegressor__ is class of this model.


## Random Forest Regression
This model is like decision tree regression.

### Functions:
__RandomForestRegressor__ is class of this model.


# Regression Template
In this folder, there are some template to use for regression models.


# Classification


## Logistic Regression
In this part we do same as regression and need feature scaling.

### Functions:
__LogisticRegression__ is class of this model.

__confusion_matrix__ is used to show a matrix to show us the accuracy.

__accuracy_score__ is used to show the accuracy.


## K Nearest Neighbor
This is as same as __LogisticRegression__.

### Functions:
__KNeighborsClassifier__ is class of this model.


## Support Vector Machine
This is as same as __LogisticRegression__.

### Functions:
__SVC__ is class of this model.


## Kernel SVM
This is just like __Support Vector Machine__ and, we need just to put rbf in the __kernel__ of the class.


## Naive Bayes
This is as same as __LogisticRegression__.

### Functions:
__GaussianNB__ is class of this model.


## Decision Tree Classification
This is as same as __LogisticRegression__.

### Functions:
__DecisionTreeClassifier__ is class of this model.


## Random Forest Classification
This is as same as __LogisticRegression__.

### Functions:
__RandomForestClassifier__ is class of this model.


# Classification Template
In this folder, there are some template to use for classification models.


# Clustering

## K Means Clustering
In this file we try to cluster data by k-means++ and have a visualising on the clustering.

### Functions:
__KMeans__ is class of this model. Here we need to initial the init by k-means++.

__kMeans.fit_predict__ is the function that cluster data.


## Hierarchical Clustering

### Libraries:
In here we use __Scipy__ to use __dendrogram__ function.

### Functions:
__dendrogram__ is used to for dendrogram plotting.

__AgglomerativeClustering__ is class if this model.


# Association Rules Learning


## Apriori
In this file we will recommend products.

### Libraries:
In here we use __apyori__ to use __apriori__ function.

### Functions:
__apriori__ is used to find the rules. For this function we need to fix __transactions, min_support, min_confidence, min_lift, min_length, max_length__.


## Eclat
This is as same as __Apriori__.


# Reinforcement


## Upper Confidence Bound
In here we calculate the reward by three steps of algorithm.


## Thompson Sampling
In here we calculate the reward by three steps of algorithm. This model is exactly more powerful than UCB.


# Natural Language Processing
In this section we try to create a model to understand English.

### Libraries:
In here we use __re__ and __nltk__ for cleaning the data.

### Functions:
__CountVectorizer__ is used to create the __Bag of Word__ model.


# Deep Learning


## Artificial Neural Network
In this section we classify by neural network.

### Libraries:
__tensorflow__ is the best library for deep learning.

### Functions:
__keras.models.Sequential__ is used to make the **Neural Network**.

__keras.layers.Dense__ is used to add layer to **Neural Network**.

__compile__ is used to configure the model for training.


# Dimensionality Reduction


## Principal Component Analysis
In this part we reduce the number of feature to reduce the complexity of the dataset.

### Functions:
__PCA__ is used to reduce the dimensions.


## Linear Discriminant Analysis
In this part we reduce the number of feature to reduce the complexity of the dataset.

### Functions:
__LinearDiscriminantAnalysis__ is used to reduce the dimensions.


## Kernel Principal Component Analysis
In this part we reduce the number of feature to reduce the complexity of the dataset.

### Functions:
__KernelPCA__ is used to reduce the dimensions.