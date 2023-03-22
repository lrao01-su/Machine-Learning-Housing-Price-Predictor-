import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import matplotlib.pyplot as plt         # pip install matplotlib
import streamlit as st                  # pip install streamlit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from pages.B_Preprocess_Data import remove_nans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import random
import plotly.express as px
random.seed(10)
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 2 - Predicting Housing Prices Using Regression")

#############################################

st.title('Train Model')

#############################################

# Complete this helper function from HW1 
def split_dataset(X, y, number,random_state=45):
    """
    This function splits the dataset into the train data and the test data

    Input: 
        - X: training features
        - y: training targets
        - number: the ratio of test samples
    Output: 
        - X_train: training features
        - X_val: test/validation features
        - y_train: training targets
        - y_val: test/validation targets
    """
    X_train = []
    X_val = []
    y_train = []
    y_val = []
    try:
        # Split data into train/test split
        # Add code here 
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=number/100, random_state=random_state)
        train_percentage = len(X_train)/(len(X_train)+len(X_val))*100
        test_percentage = len(X_val)/(len(X_train)+len(X_val))*100

        # Print dataset split result
        st.markdown('The training dataset contains {0:.2f} observations ({1:.2f}%) and the test dataset contains {2:.2f} observations ({3:.2f}%).'.format(len(X_train),
                                                                                                                                                          train_percentage,
                                                                                                                                                          len(X_val),
                                                                                                                                                          test_percentage))
        # Save state of train and test splits in st.session_state
        st.session_state['X_train'] = X_train
        st.session_state['X_val'] = X_val
        st.session_state['y_train'] = y_train
        st.session_state['y_val'] = y_val
    except:
        print('Exception thrown; testing test size to 0')
    return X_train, X_val, y_train, y_val

# Checkpoint 7
def inspect_coefficients(models, inspect_models):
    """
    This function gets the coefficients of the trained models

    Input: 
        - models: all trained models
        - inspect_models: the models to be inspected on
    Output: 
        - out_dict: a dicionary contains the coefficients of the selected models, with the following keys:
            - 'Multiple Linear Regression'
            - 'Polynomial Regression'
            - 'Ridge Regression'
            - 'Lasso Regression'
    """
    out_dict = {'Multiple Linear Regression': [],
                'Polynomial Regression': [],
                'Ridge Regression': [],
                'Lasso Regression': []}
    
    # Add code here
    for model_name in inspect_models:
        if(model_name in models):
            current_model = models[model_name]
            if(model_name == 'Multiple Linear Regression'):
                out_dict['Multiple Linear Regression'] = current_model.coef_
                st.write(out_dict['Multiple Linear Regression'])
            elif(model_name == 'Polynomial Regression'):
                out_dict['Polynomial Regression'] = current_model.named_steps['polyReg'].coef_
                st.write(out_dict['Polynomial Regression'])
            elif(model_name == 'Ridge Regression'):
                out_dict['Ridge Regression'] = current_model[-1].best_estimator_.coef_
                st.write(current_model[-1].cv_results_)
            elif(model_name == 'Lasso Regression'):
                out_dict['Lasso Regression'] = current_model[-1].best_estimator_.coef_
                st.write(current_model[-1].cv_results_)    
    return out_dict
                

# Checkpoint 8
def train_multiple_regression(X_train, y_train, regression_methods_options):
    """
    Fit a multiple regression model to data 

    Input: 
        - X_train: training features
        - y_train: training targets
        - regression_methods_options: a list of all possible model names
    Output: 
        - multi_reg_model: the trained multiple regression model
    """
    multi_reg_model=None

    # Train model. Handle errors with try/except statement
    # Add code here
    if 'Multiple Linear Regression' in regression_methods_options:
        try:
            multi_reg_model = LinearRegression()
            multi_reg_model.fit(X_train, y_train)
        except:
            st.write('Error: Multiple Linear Regression model failed to train.')
    # Save state of model in st.session_state
    st.session_state['Multiple Linear Regression'] = multi_reg_model
    return multi_reg_model

# Checkpoint 8
def train_polynomial_regression(X_train, y_train, regression_methods_options, poly_degree, poly_include_bias):
    """
    This function trains the model with polynomial regression

    Input: 
        - X_train: training features
        - y_train: training targets
        - regression_methods_options: a list of all possible model names
        - poly_degree: the degree of polynomial
        - poly_include_bias: whether or not to include bias
    Output: 
        - poly_reg_model: the trained model
    """
    poly_reg_model=None

    # Train model. Handle errors with try/except statement
    # Add code here
    if 'Polynomial Regression' in regression_methods_options:
        try:
            poly_reg_model = Pipeline([('scaler', StandardScaler()), 
                           ('poly', PolynomialFeatures(degree=poly_degree, include_bias=poly_include_bias)), 
                           ('polyReg', LinearRegression())])
            poly_reg_model.fit(X_train, y_train)
            #print("Coefficients:", poly_reg_model.named_steps['polyReg'].coef_)
        except:
            st.write('Error: Polynomial Regression model failed to train.')
    # store model in st.session_state[model_name]
    st.session_state['Polynomial Regression'] = poly_reg_model
    return poly_reg_model

# Checkpoint 8
def train_ridge_regression(X_train, y_train, regression_methods_options, ridge_params, ridge_cv_fold):
    """
    This function trains the model with ridge regression and cross-validation

    Input: 
        - X_train: training features
        - y_train: training targets
        - regression_methods_options: a list of all possible model names
        - ridge_params: a dictionary of the hyperparameters to tune during cross validation
        - ridge_cv_fold: the number of folds for cross validation
    Output: 
        - ridge_cv: the trained model
    """
    ridge_cv=None
    # Train model with ridgeregression and cross-validation. Handle errors with try/except statement
    # Add code here
    #train with ridge regression and cross validation
    if 'Ridge Regression' in regression_methods_options:
        try:
            ridge_cv = Pipeline([('scaler', StandardScaler()), 
                     ('ridgeCV', GridSearchCV(estimator=Ridge(), 
                     param_grid=ridge_params, cv=ridge_cv_fold))])
            ridge_cv.fit(X_train, y_train)
            #print("Coefficients:", ridge_cv[-1].best_estimator_.coef_)
        except:
            #print input parameters
            st.write('Error: Ridge Regression model failed to train.')
    st.session_state['Ridge Regression'] = ridge_cv
    return ridge_cv

# Checkpoint 8
def train_lasso_regression(X_train, y_train, regression_methods_options, lasso_params, lasso_cv_fold):
    """
    This function trains the model with lasso regression and cross-validation

    Input: 
        - X_train: training features
        - y_train: training targets
        - regression_methods_options: a list of all possible model names
        - lasso_params: a dictionary of the hyperparameters to tune during cross validation
        - lasso_cv_fold: the number of folds for cross validation
    Output: 
        - lasso_cv: the trained model
    """
    lasso_cv=None
    
    # Train model. Handle errors with try/except statement
    # Add code here
    if 'Lasso Regression' in regression_methods_options:
        try:
            lasso_cv = Pipeline([('scaler', StandardScaler()), ('lassoCV', GridSearchCV(estimator=Lasso(), param_grid=lasso_params, cv=lasso_cv_fold))])
            lasso_cv.fit(X_train, y_train)
            #print("Coefficients:", lasso_cv[-1].best_estimator_.coef_)
        except:
            st.write('Error: Lasso Regression model failed to train.')
    # store model in st.session_state[model_name]
    st.session_state['Lasso Regression'] = lasso_cv
    #st.write('train_lasso_regression not implemented yet.')
    return lasso_cv

# Helper function
@st.cache
def convert_df(df):
    """
    Cache the conversion to prevent computation on every rerun

    Input: 
        - df: pandas dataframe
    Output: 
        - Save file to local file system
    """
    return df.to_csv().encode('utf-8')

###################### FETCH DATASET #######################
# df = ... Add code here: read in data and store in st.session_state
def restore_dataset():
    """
    Input: 
    Output: 
    """
    df=None
    if 'house_df' not in st.session_state:
        data = st.file_uploader('Upload a Dataset', type=['csv', 'txt'], key='dataset_c')
        if data:
            df = pd.read_csv(data)
            st.session_state['house_df'] = df
        else:
            return None
    else:
        df = st.session_state['house_df']
    return df
df = restore_dataset()
if df is not None:
    # Display dataframe as table
    st.dataframe(df.describe())

    # Select variable to predict
    feature_predict_select = st.selectbox(
        label='Select variable to predict',
        options=list(df.select_dtypes(include='number').columns),
        key='feature_selectbox',
        index=8
    )

    st.session_state['target'] = feature_predict_select

    # Select input features
    feature_input_select = st.multiselect(
        label='Select features for regression input',
        options=[f for f in list(df.select_dtypes(
            include='number').columns) if f != feature_predict_select],
        key='feature_multiselect'
    )

    st.session_state['feature'] = feature_input_select

    st.write('You selected input {} and output {}'.format(
        feature_input_select, feature_predict_select))

    df = remove_nans(df)
    X = df.loc[:, df.columns.isin(feature_input_select)]
    Y = df.loc[:, df.columns.isin([feature_predict_select])]

    # Split train/test
    st.markdown('## Split dataset into Train/Test sets')
    st.markdown(
        '### Enter the percentage of test data to use for training the model')
    number = st.number_input(
        label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

    # Compute the percentage of test and training data
    X_train, X_val, y_train, y_val = split_dataset(X, Y, number)

    regression_methods_options = ['Multiple Linear Regression',
                                  'Polynomial Regression', 'Ridge Regression', 'Lasso Regression']
    # Collect ML Models of interests
    regression_model_select = st.multiselect(
        label='Select regression model for prediction',
        options=regression_methods_options,
    )
    st.write('You selected the follow models: {}'.format(
        regression_model_select))

    ###################### TRAIN REGRESSION MODELS #######################
    # # Add parameter options to each regression method
    
    # Multiple Linear Regression
    if (regression_methods_options[0] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[0])
        if st.button('Train Multiple Linear Regression Model'):
            train_multiple_regression(
                X_train, y_train, regression_methods_options)
        if regression_methods_options[0] not in st.session_state:
            st.write('Multiple Linear Regression Model is untrained')
        else:
            st.write('Multiple Linear Regression Model trained')

    # Polynomial Regression
    if (regression_methods_options[1] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[1])

        poly_degree = st.number_input(
            label='Enter the degree of polynomial',
            min_value=0,
            max_value=10,
            value=3,
            step=1,
            key='poly_degree_numberinput'
        )
        st.write('You set the polynomial degree to: {}'.format(poly_degree))

        poly_include_bias = st.checkbox('include bias')
        st.write('You set include_bias to: {}'.format(poly_include_bias))

        if st.button('Train Polynomial Regression Model'):
            train_polynomial_regression(
                X_train, y_train, regression_methods_options, poly_degree, poly_include_bias)

        if regression_methods_options[1] not in st.session_state:
            st.write('Polynomial Regression Model is untrained')
        else:
            st.write('Polynomial Regression Model trained')

    # Ridge Regression
    if (regression_methods_options[2] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[2])
        ridge_cv_fold = st.number_input(
            label='Enter the number of folds of the cross validation',
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            key='ridge_cv_fold_numberinput'
        )
        st.write('You set the number of folds to: {}'.format(ridge_cv_fold))

        solvers = ['auto', 'svd', 'cholesky', 'lsqr',
                   'sparse_cg', 'sag', 'saga', 'lbfgs']
        ridge_solvers = st.multiselect(
            label='Select solvers for ridge regression',
            options=solvers,
            default=solvers[0],
            key='ridge_reg_solver_multiselect'
        )
        st.write('You select the following solver(s): {}'.format(ridge_solvers))

        ridge_alphas = st.text_input(
            label='Input a list of alpha values, separated by comma',
            value='1.0,0.5',
            key='ridge_alphas_textinput'
        )
        st.write('You select the following alpha value(s): {}'.format(ridge_alphas))

        ridge_params = {
            'solver': ridge_solvers,
            'alpha': [float(val) for val in ridge_alphas.split(',')]
        }

        if st.button('Train Ridge Regression Model'):
            train_ridge_regression(
                X_train, y_train, regression_methods_options, ridge_params, ridge_cv_fold)

        if regression_methods_options[2] not in st.session_state:
            st.write('Ridge Model is untrained')
        else:
            st.write('Ridge Model trained')

    # Lasso Regression
    if (regression_methods_options[3] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[3])
        lasso_cv_fold = st.number_input(
            label='Enter the number of folds of the cross validation',
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            key='lasso_cv_fold_numberinput'
        )
        st.write('You set the number of folds to: {}'.format(lasso_cv_fold))

        lasso_tol = st.text_input(
            label='Input a list of tolerance values, separated by comma',
            value='0.001,0.0001',
            key='lasso_tol_textinput'
        )
        st.write('You select the following tolerance value(s): {}'.format(lasso_tol))

        lasso_alphas = st.text_input(
            label='Input a list of alpha values, separated by comma',
            value='1.0,0.5',
            key='lasso_alphas_textinput'
        )
        st.write('You select the following alpha value(s): {}'.format(lasso_alphas))

        lasso_params = {
            'tol': [float(val) for val in lasso_tol.split(',')],
            'alpha': [float(val) for val in lasso_alphas.split(',')]
        }

        if st.button('Train Lasso Regression Model'):
            train_lasso_regression(
                X_train, y_train, regression_methods_options, lasso_params, lasso_cv_fold)

        if regression_methods_options[3] not in st.session_state:
            st.write('Lasso Model is untrained')
        else:
            st.write('Lasso Model trained')

    ###################### INSPECT MODEL COEFFICIENTS #######################
    st.markdown('## Inspect model coefficients')

    # Select multiple models to inspect
    inspect_models = st.multiselect(
        label='Select features for regression input',
        options=regression_model_select,
        key='inspect_multiselect'
    )
    
    models = {}
    for model_name in inspect_models:
        models[model_name] = st.session_state[model_name]
    coefficients = inspect_coefficients(models, inspect_models)
    #store dataset in session state
    st.session_state['coefficients'] = coefficients
    


    ###################### DOWNLOAD DATASET #######################
    st.markdown('### Download the dataset')

    csv = convert_df(df)

    st.write('Saving dataset to paml_dataset.csv')
    st.download_button(
        label="Train: Download data as CSV",
        data=csv,
        file_name='paml_dataset.csv',
        mime='text/csv',        
    )

    st.write('Continue to Test Model')