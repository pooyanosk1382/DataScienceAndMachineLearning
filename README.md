# DataScienceAndMachineLearning
Codes of machine learning course and summary of each lesson.

## Data preprocesssing in python:

### Library:
There are some useful libraries such as numpy, matplotlib, pandas and the most useful sklearn.

### Function:
__SimpleImputer__ use to take care of missing data.

__OneHotEncoder__ changes strings to vectors.

__ColumnTransformer__ changes columns of dataset.

__LabelEncoder__ is used to convert two-state data to 0 and 1.

__train_test_split__ splits independent and dependent variable to two entity. Trian and test. 

__StandardScaler__ is used to apply standardisation on data.


## Simple Linear Regression:

### Function:
__LinearRegression__ is a object to fit a regression on a train set.

__.predict__ is a function to predict the y by use of regression object.

## Multiple Linear Regression
The functions is just like __Simple Linear Regression__ just need to encode the non-numberical column by OneHotEncoder.

__Tip1:__ The LinearRegression function handle dummy variable and choose best feature.

__Tip2:__ Because the number of elements are more than two we can't show them by graphs so we just check the prediction with test set.
