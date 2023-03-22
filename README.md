[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10374688&assignment_repo_type=AssignmentRepo)
# Practical Applications in Machine Learning 

# Homework 2: Predict Housing Prices Using Regression

The goal of this assignment is to build a regression machine learning pipeline in a web application to use as a tool to analyze the models to gain useful insights about model performance. Note, that the ‘Explore Dataset’ and ‘Preprocess Data’ steps from homework 1 can be re-used for this assignment.

The <b>learning outcomes</b> for this assignment are:
* Build end-to-end regression pipeline using 1) multiple regression, 2) polynomial regression, and 3) ridge, and 4) lasso regression.
* Evaluate regression methods using standard metrics including root mean squared error (RMSE), mean absolute error (MAE), and coefficient of determination (R2).
* Develop a web application that walks users through steps of the regression pipeline and provide tools to analyze multiple methods across multiple metrics. 
* Develop a web application that offers a service to customers by predicting housing prices using regression models. 

# Assignment Deadline:
* Due:  Friday March 18, 2023 at 11:00PM 
* What to turn in: Submit responses on GitHub AutoGrader
* Assignment Type: Individual
* Time Estimate: 18 Hours
* Submit code via GitHub: https://classroom.github.com/a/7LmoWBZw
* Submit Reflection Assessment via Canvas (7 multiple choice questions)

## Assignment Outline
* Setup
* End-to-End Regression Models
* Testing Code with Github Autograder
* Reflection Assessment

# Reading Prerequisites 

* Review the jupyter notebook in Chapter 4 Training Models of “Machine Géron, Aurélien. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow.” O’Reilly Media, Inc., 2022. (Available on Canvas->Library Reserves). We will build on that example.
* Review [HW2 Web App Design Slides](https://cornell.box.com/s/dgt5gpp6kiu58n6u6jg24y4k6aqcotfp).

# Computing Prerequisites

* [Optional] Revisit the [Preliminaries Github repository](https://github.com/Cornell-Tech-PAML-Course/0-paml-preliminaries) if you have not set up your computing environment. 
* Clone Homework 2 Github using the [Github Autograder Invitation Link](https://classroom.github.com/a/7LmoWBZw).

# End-to-End Regression Models for Housing Prices 

## California Housing Data

This assignment involves testing the end-to-end pipeline in a web application using a California Housing dataset from the textbook: Géron, Aurélien. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow. O’Reilly Media, Inc., 2022. The dataset was captured from California census data in 1990. We have added additional features to the dataset. The features include:
* longitude - longitudinal coordinate
* latitude - latitudinal coordinate
* housing_median_age - median age of district
* total_rooms - total number of rooms per district
* total_bedrooms - total number of bedrooms per district
* population - total population of district
* households - total number of households per district'
* median_income - median income
* ocean_proximity - distance from the ocean
* median_house_value - median house value
* city - city location of house
* county - county of house
* road - road of the house
* postcode - zip code 

The Github repository contains two datasets in the dataset directory:
* housing - dataset from the textbook and Kaggle 
* housing_paml_hw2 - modified dataset for HW2. Use this dataset for testing HW1.

# Explore and Visualize Data (3 points)

Your goal is to build on the system you developed from homework 1 and additional options to explore datasets.

## Task 1: Import or restore a dataset and display it 

Your goal is to create two columns 1) upload a dataset from a file on your and 2) upload a dataset from a cloud source (specifically for HW2). Then, store the dataset in streamlit session state for later use. This task is repeated for all pages on the web application except Pages A_Explore Dataset and D_Deploy App. On Pages B - D, make sure to restore the dataset from st.session_state to avoid importing multiple times. Feel free to reuse code from HW1.

## Task 2: Generate and show visualizations of features in the dataset

Generate and show visualizations of features in the dataset using a sidebar as done in HW1 (re-use code). 

## Task 3 (Checkpoint 1): Show feature correlations

Write a function compute_correlation() that computes the correlation between two features specified by a user. Recall: Correlation is a quantitative measure of how much two variables/features are correlated ranging from -1 (negatively correlated) to 1 (positively correlated). Feel free to reuse code from HW1. In addition, include summary statements for each pair of features stating the magnitude and direction of correlation. For example, ‘Features median_house_age and median_housing_price are highly positively correlated.’ See the HW2 code for inputs and outputs.

# Preprocess and Prepare Data for Regression (8 points)

Your goal is to build on the system you developed from HW1 and additional options to preprocess data. 

## Task 4 (Checkpoint 2): Summarize missing or invalid data

Use Pandas function isna() to find the missing values in the dataset. Write a function called summarize_missing_data() that computes four statistics:

* Number of categories with missing values.
* Average number of missing values per category. 
* Total number of missing values.
* Categories with most top N missing values where N is a number greater than 1.  

Feel free to reuse code from HW1 and the [‘Predicting Housing Prices’ notebook](https://github.com/Cornell-Tech-PAML-Course/predicting-housing-prices). See the HW2 code for inputs and outputs.

Task 5 and 6: Add two columns to the web application using st.columns() function, one for Task 6 and 7. 

## Task 5: Remove irrelevant/useless features

Collect a user's preferences on one or more features to remove using the st.multiselect() function. Add a button to remove irrelevant/useless values using the pd.drop() function. Feel free to reuse code from HW1.

## Task 6 (Checkpoint 3): Remove Nans

Add a button to remove ‘Not a Number’ (Nan) values from the dataset which calls the remove_nans() using the pd.dropna()  function. Write a function called remove_nans() that returns a dataframe with Nans removed.

## Task 7: Impute data 

Impute the dataset with one of the following options: 1) mean, 2) median, 3) zeros. Collect a user’s preference for one feature to impute imputation method (from previous sentence). Next, show the updated dataset showing the imputed datapoint updates. 

## Task 8 (Checkpoint 4): Remove outliers

Remove outliers from the dataset using the InterQuartile Range (IQR), a commonly used approach for finding and removing outliers.

Display a plot, as done in Task 2 to view a Scatter Plot, Boxplot, Line Plot, or Histogram to inspect features before removing outliers. Add a st.multiselect() menu with the numerical features as options using the code below, given that df is the Pandas dataframe containing the dataset. 
```
numeric_columns = list(df.select_dtypes(include='number').columns)
```

Then, add a button that calls the remove_outliers function which takes a pandas dataframe (df) and feature (string) as input and returns that dataset with outliers removed from the feature. This function computes the 25th (lower bound) and 75th percentile (upper bound), uses the upper and lower bounds to filter the feature, and returns the updated dataframe. Lastly, the function should display the number of outliers removed from the dataset. For example, ‘Outliers for feature median_income are lower than 2543 and higher than 28374.’ See example in the ‘Predicting Housing Prices’ notebook.

Write a function called remove_outliers() using the following function(s) (See the HW2 code for inputs and outputs):
pd.dropna() - function that drops rows or columns with Nan values. Drop Nan values before removing outliers.
np.percentile() - function that computes the q-th percentile of data along the specified axis. Using this function to compute the 25th and 75th percentile.

## Task 9 (Checkpoint 5): Handling Text/Categorical Data 

Encode categorical features with integer encoding and one-hot encoding and write one function for each encoding technique. Add a button for each method which calls the functions below for their respective functions.

Write a function called integer_encode_feature() using pd.get_dummies() function - converts categorical variables to indicator function (i.e., one-hot encoding). Then, use st.button() to create a button to run the function. See the HW2 code for inputs and outputs.

Write a function called one_hot_encode_feature() using OrdinalEncoder fit_transform() - maps categorical variables to unique integers (i.e., integer encoding). Then, use st.button() to create a button to run the function. See the HW2 code for inputs and outputs.

## Task 10 (Checkpoint 6): Create new features

Create new features using existing ones in the dataset using four mathematical operations including 1) addition, 2) subtraction, 3) multiplication, 4) division, and 5) square root, 6) ceil, and 7) floor. Collect two numerical features for the math operations using st.multiselect(), except square root which takes one feature as input using st.selectbox(). See example for collecting numerical columns below:  
```
numeric_columns = list(df.select_dtypes(include='number').columns)
```
Write the create_feature() function that creates a new feature using the output. It takes a pandas dataframe, a list of mathematical operations to apply to one or more features, and a list of one (for square root) or two features (all other features). For example, a user can select to ‘add’ total_bedrooms and total_rooms then the function adds these two features, and names the feature with the feature name specified as input. See the HW2 code for inputs and outputs.

Use the following function(s) for create_feature():
* np.sqrt() - function that computes the square root of a number. Compute the square root of a feature when selected by the user.
* np.ceil() - rounds float values up
* np.floor() - rounds float values down 

<b>Task 11</b>: Summarize descriptive statistics. Feel free to reuse code from HW1.  

# Train Regression Models (7 points)

Set a fixed seed of 10 for all regression models (see example below).
```
import random 
random.seed(10)
```

## Task 12: Set train/test split using your code from HW1. 

In addition, write a function called split_dataset() that splits the dataset into four parts, input and output training and test sets (e.g., X_train, X_val, y_train, y_val)using the following function(s):
number_input(): function that displays a text field and a user to input a number. Collect the test set size from the text field.
Sklearn train_test_split() - function that splits that dataset into training and testing sets. Split the dataset into input X and output Y values for a training and test set.

<b>Task 13</b>: Select model input and output variables, and model parameters. Recall: Regression aims to predict output variable Y from input variable(s) X. Your goal is to set the input and output variables with input from a user using pandas filtering (see example below). The input variable can contain one or more features and the output variable contains one feature to predict (i.e., house prices). 
```
A = df.loc[:, df.columns.isin(feature)] # collects columns with the feature name
B = df.loc[:, ~df.columns.isin(feature)] # collects columns without the feature name
```

## Task 14: Train Multiple Regressoon Model

Write a function called train_multiple_regression() that trains a multiple regression regression model to predict an output variable (i.e., housing prices). Add a button to call the train_multiple_regression() function and train the model using Sklearn library: linear_model.LinearRegression(). See the HW2 code for inputs and outputs.

## Task 15: Train Polynomial Regressoon Model

Write a function called train_polynomial_regression() that trains a polynomial regression regression model to predict an output variable (i.e., housing prices). Then, display the training performance in terms of the metrics selected by users. Create a parameter input panel for each method that requires model parameters including degrees and include_bias.  Then, add a button to call the train_polynomial_regression() function and train the model with Sklearn library: linear_model, preprocessing.PolynomialFeatures or Pipeline. See the HW2 code for inputs and outputs.

## Task 16: Train Ridge Regressoon Model with Cross Validation

Write a function called train_ridge_regression that uses cross validation to train a ridge regression regression model to predict housing prices and display the training performance in terms of the metrics selected by users. Create a parameter input panel for each method that requires model parameters alpha, solver ('auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'), and n_folds.  Then, add a button to call the train_ridge_regression function and train the model using Sklearn library: linear_model.Ridge() or Pipeline. See the HW2 code for inputs and outputs.

## Task 17: Train Polynomial Regressoon Model

Write a function called train_lasso_regression that uses cross validation (CV) to train a lasso regression regression model to predict housing prices and display the training performance in terms of the metrics selected by users. Create a parameter input panel for each method that requires input parameters including tol, alpha, n_folds.  Then, add a button to call the train_lasso_regression function and train the model using Sklearn library: linear_model.Lasso() or Pipeline. See the HW2 code for inputs and outputs.

## Task 18 (Checkpoint 7): Inspect Regression coefficients

Write a function called inspect_coefficients() that takes all trained regression models as input and returns their respective coefficients. The Sklearn library implements both closed-form and gradient descent solutions to generate regression coefficients. The coefficients can be inspected using the coef_ and intercept_ attributes (see example below). Once one or more regression models have been trained, display the regression coefficients to be used for inspection (use for reflection assessment). Use st.multiselect menu with features as options - used to select features for regression input.

# Test Regression Models (4 points)

## Task 19: Evalaute models with performance metrics.

Provide users with the option to select up to three metrics for evaluation and with the option to select multiple trained models:

* Root mean squared error (RMSE) measures the difference between predicted and actual values using Euclidean distance. Use Sklearn mean_squared_error() function that computes mean squared error (squared=True) and root mean squared error (squared=False).
* Mean absolute error (MAE) measures the absolute difference between predicted and actual values. Use Sklearn mean_absolute_error() function that computes mean absolute error.
* Coefficient of determination (R2) represents proportion of variance in predicted values that can be explained by the input features. Furthermore, it represents a goodness of fit, thereby, how well the model can generalize to new examples. Use Sklearn * r2_score() function that computes the coefficient of determination (R2).

Write a function called calculate_metrics() that takes the predicted and actual housing values as input and returns the metrics for each method using the following: See the HW2 code for inputs and outputs.

## Task 20 (Checkpoint 8): Plot Learning Curves

Write a function plot_learning_curve() that displays figures summarizing learning curves that show predicted housing prices vs the training set size (error vs. training set size). See the HW2 code for inputs and outputs.

Use the following functions in plot_learning_curve()
* st.plotly_chart() - used to display learning curves. 
* Plotly.subplots make_plots() function to create learning curve plots. Use the fig.add_trace() function to plot training and validation curves. 

## Task 21 (Checkpoint 9): Display Evaluation Results.

Write a function called compute_eval_metrics() to display tables summarizing the results in terms of evaluation metrics by regression model for selected metrics. This function computes the RMSE, MAE, and R2 score for the selected model and metrics from Task 20. Your goal is to populate the metricl_dict with the metric name as a key and the computed results as values. See the HW2 code for inputs and outputs.

## Task 22 (Checkpoint 10): Deploy Model (1 point)

Before deploying the application, train regression models using the following features: number_bedrooms, number_bathrooms, and ocean_proximity. Then, deploy a regression model with best performance out of the four models implemented including multiple regression, polynomial regression, and ridge regression, and lasso regression for a given metric. Provide the user the option to select a regression model based on the evaluation results to use for deployment performance using the st.selectbox() function to create a menu for selecting a single item. 

# Deploy ML Application (3 points)

Create a California Housing Price application that collects a user's housing preferences to use for predicting housing prices. Once a user provides their preferences, deploy a regression model with the best performance given a pre-defined metric from Task 19.

## Task 23 (Checkpoint 11): Deploy Model in Housing Application

Write a function called deploy_model() which tests a regression model with best performance out of all regression models for a given metric. Use the deployment model selected on the ‘Test Model’ page to train the specified regression model and hyperparameters.  The next element shown on the web application is a map of California showing the available house listings.

The web application will show the predicted housing price and state whether they can afford it based on their budget.

$ Testing Code with Github Autograder

Test your homework solution as needed using Github Autograder. Clone your personal private copy of the homework assignment. Once the code is downloaded and stored on your computer in your directory of choice, you can start testing the assignment. To test your code, open a terminal and navigate to the directory of the homework2.py file. Then, type ‘pytest’ ad press enter. The autograder with print feedback to inform you what checkpoint functions are failing. Test homework2.py using this command in your terminal:

```
streamlit run homework2.py
```

# Reflection Assessment

Complete on Canvas.





