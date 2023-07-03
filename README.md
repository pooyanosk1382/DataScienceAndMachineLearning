# DataScienceAndMachineLearning
This repository contains codes and summary of each lesson of the machine learning course.


# Data preprocessing in python:


### Library:
There are some useful libraries such as numpy, matplotlib, pandas and the most useful **sklearn**.

### Function:
__SimpleImputer__ is used to take care of missing data.

__OneHotEncoder__ changes strings to numerical vectors.

__ColumnTransformer__ changes columns of dataset.

__LabelEncoder__ is used to convert two-state data to 0 and 1.

__train_test_split__ will split the whole dataset to two main parts, **Train & Test**.

__StandardScaler__ is used to apply standardization on data.


# Regression


## Simple Linear Regression:

### Function:
__LinearRegression__ is an object to fit a regression on a train set.

__.predict__ is a function to predict the y by use of regression object.

## Multiple Linear Regression
The functions are just like __Simple Linear Regression__ just need to encode the non-numerical column by OneHotEncoder.

__Tip1:__ The LinearRegression function handle dummy variable and choose best feature.

__Tip2:__ Because the number of elements are more than two we can't show them by graphs ,so we just check the prediction with test set.


## Polynomial Regression

For this part we do just as same the multiple linear regression just create x^2 and x^3 and ... by PolynomialFeatures.

### Function:
__PolynomialFeatures__ is used to create x and x^2and x^3 and ...


## Support Vector Regression
The first thing to mention is that we use feature scaling on both x and y because this is how svr works.

### Function:
__SVR__ that is a class ,and we should make an object from this.

__inverse_transform__ that is used to return the values to the state before feature scaling


## Decision Tree Regression
This model is not good for one feature and is better to use it when we have some features. This doesn't need feature scaling. Other things are same as last parts.

### Function:
__DecisionTreeRegressor__ is class of this model.


## Random Forest Regression
This model is like decision tree regression.

### Function:
__RandomForestRegressor__ is class of this model.


# Regression Template
In this folder, there are some template to use for regression models.


# Classification


## Logistic Regression
In this part we do same as regression and need feature scaling.

### Function:
__LogisticRegression__ is class of this model.

__confusion_matrix__ is used to show a matrix to show us the accuracy.

__accuracy_score__ is used to show the accuracy.


## K Nearest Neighbor
This is as same as __LogisticRegression__.

### Function:
__KNeighborsClassifier__ is class of this model.


## Support Vector Machine
This is as same as __LogisticRegression__.

### Function:
__SVC__ is class of this model.


## Kernel SVM
This is just like __Support Vector Machine__ and, we need just to put rbf in the __kernel__ of the class.


## Naive Bayes
This is as same as __LogisticRegression__.

### Function:
__GaussianNB__ is class of this model.


## Decision Tree Classification
This is as same as __LogisticRegression__.

### Function:
__DecisionTreeClassifier__ is class of this model.