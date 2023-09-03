"""MACHINE LEARNING PIPELINES
Final Pipeline
Way to go! Now that we are getting the hang of pipelines, we’re going take things up a notch. We will now be searching over different types of models, each having their own sets of hyperparameters! In the original pipeline, we defined regr to be an instance of LinearRegression(). Then in defining the parameter grid to search over, we used the dictionary {"regr__fit_intercept": [True,False]} to define the values of the fit_intercept term. We can equivalently do this by passing both the estimator AND parameters in a single dictionary as

{'regr': [LinearRegression()], "regr__fit_intercept": [True,False]}

We can add more models to it as follows. Suppose we wanted to add a Ridge regression model and also perform hyperparamter tuning using GridSearchCV to find the best regularization parameter alpha, we would add it to previous dictionary to create an array of dictionaries as follows:

search_space = [{'regr': [LinearRegression()], 'regr__fit_intercept': [True,False]},
                {'regr':[Ridge()], 'regr__alpha': [0,0.1,1,10,100]}

Note: If you’d like a refresher on regularization using hyperparameter tuning in regression models, check out our article and lesson on the same.

The goal of this process is to find the best estimator for our dataset and problem in the most efficient manner possible. The pipeline module allows us to do exactly that! In a couple of lines of code, we’re able to preprocess the data and search an entire model and hyperparameter space. The final step is to access the pipeline elements to draw out the information about which estimator and hyperparameter set gets us the best score. We do this by using the .next_steps method by using the strings we’ve used in the dictionary. For instance, the regression model can be access using the string 'regr' from the dictionary as follows:

Get the best estimator using GridSearchCV‘s .best_estimator_ method
Use .named_steps['regr'].get_params() on the best estimator to get its hyperparameters!"""

import numpy as np
import pandas as pd

from sklearn import svm, datasets
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn import metrics

columns = ["sex","length","diam","height","whole","shucked","viscera","shell","age"]
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",names=columns)

y = df.age
X=df.drop(columns=['age'])
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include=['object']).columns
#create some missing values
for i in range(1000):
    X.loc[np.random.choice(X.index),np.random.choice(X.columns)] = np.nan

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25)

cat_vals = Pipeline([("imputer",SimpleImputer(strategy='most_frequent')), ("ohe",OneHotEncoder(sparse=False, drop='first'))])
num_vals = Pipeline([("imputer",SimpleImputer(strategy='mean')), ("scale",StandardScaler())])

preprocess = ColumnTransformer(
    transformers=[
        ("cat_preprocess", cat_vals, cat_cols),
        ("num_preprocess", num_vals, num_cols)
    ]
)
#Create a pipeline with preprocess and a linear regression model
pipeline = Pipeline([("preprocess",preprocess), 
                     ("regr",LinearRegression())])

#--------------------------------------------------------------
# 1. Update the `search_space` array from the narrative to add a Lasso Regression model as the third dictionary.
search_space = [{'regr': [LinearRegression()], 'regr__fit_intercept': [True,False]},
                {'regr': [Ridge()], 'regr__alpha': [0,0.1,1,10,100]},
                {'regr': [Lasso()], 'regr__alpha': [0,0.1,1,10,100]}]




# 2.  Initialize a grid search on `search_space`
gs = GridSearchCV(pipeline, search_space, scoring='neg_mean_squared_error', cv = 5)

#3. Find the best pipeline, regression model and its hyperparameters

## i. Fit to training data
gs.fit(x_train, y_train)

## ii. Find the best pipeline
best_pipeline = gs.best_estimator_

## iii. Find the best regression model
best_regression_model = best_pipeline.named_steps['regr']
print('The best regression model is:')
print(best_regression_model)

## iv. Find the hyperparameters of the best regression model
best_model_hyperparameters = best_regression_model.get_params()
print('The hyperparameters of the regression model are:')
print(best_model_hyperparameters)

#4. Access the hyperparameters of the categorical preprocessing step
cat_preprocess_hyperparameters = best_pipeline.named_steps['preprocess'].named_transformers_['cat_preprocess'].named_steps['imputer'].get_params()
print('The hyperparameters of the imputer are:')
print(cat_preprocess_hyperparameters)
