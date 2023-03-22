import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pandas.plotting import scatter_matrix
import os
import tarfile
import urllib.request
from itertools import combinations

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

feature_lookup = {
    'longitude': '**longitude** - longitudinal coordinate',
    'latitude': '**latitude** - latitudinal coordinate',
    'housing_median_age': '**housing_median_age** - median age of district',
    'total_rooms': '**total_rooms** - total number of rooms per district',
    'total_bedrooms': '**total_bedrooms** - total number of bedrooms per district',
    'population': '**population** - total population of district',
    'households': '**households** - total number of households per district',
    'median_income': '**median_income** - median income',
    'ocean_proximity': '**ocean_proximity** - distance from the ocean',
    'median_house_value': '**median_house_value**',
    'city':'city location of house',
    'road':'road of the house',
    'county': 'county of house',
    'postcode':'zip code',
    'rooms_per_household':'average number of rooms per household',
    "number_bedrooms":'number of bedrooms',
    "number_bathrooms": 'number of bathrooms',
    

}
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 2 - Predicting Housing Prices Using Regression")

#############################################

st.markdown('# Explore Dataset')

#############################################

st.markdown('### Import Dataset')

# Checkpoint 1
def compute_correlation(X, features):
    """
    This function computes pair-wise correlation coefficents of X and render summary strings 
        with description of magnitude and direction of correlation

    Input: 
        - X: pandas dataframe 
        - features: a list of feature name (string), e.g. ['age','height']
    Output: 
        - correlation: correlation coefficients between one or more features
        - summary statements: a list of summary strings where each of it is in the format: 
            '- Features X and Y are {strongly/weakly} {positively/negatively} correlated: {correlation value}'
    """
    correlation = X[features].corr()

    # Initialize an empty list to store summary statements
    summary_statements = []

    # Loop over all pairs of features and generate a summary statement for each
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            # Get the correlation coefficient
            corr = correlation.loc[features[i], features[j]]

            # Determine the strength of the correlation
            if abs(corr) >= 0.8:
                strength = 'strongly'
            else:
                strength = 'weakly'
            # Determine the direction of the correlation
            if corr >= 0:
                direction = 'positively'
            else:
                direction = 'negatively'

            # Generate the summary statement
            statement = f"- Features {features[i]} and {features[j]} are {strength} {direction} correlated: {corr:.2f}"
            # Append the summary statement to the list
            summary_statements.append(statement)

    return correlation, summary_statements

    # Add code here

# Helper Function
def user_input_features(df, chart_type, x=None, y=None):
    """
    This function renders the feature selection sidebar 

    Input: 
        - df: pandas dataframe containing dataset
        - chart_type: the type of selected chart
        - x: features
        - y: targets
    Output: 
        - dictionary of sidebar filters on features
    """
    side_bar_data = []
    select_columns = []
    if (x is not None):
        select_columns.append(x)
    if (y is not None):
        select_columns.append(y)
    if (x is None and y is None):
        select_columns = list(df.select_dtypes(include='number').columns)
    for idx, feature in enumerate(select_columns):
        try: 
            f = st.sidebar.slider(
                str(feature),
                float(df[feature].min()),
                float(df[feature].max()),
                (float(df[feature].min()), float(df[feature].max())),
                key = chart_type+str(idx)
            )
        except Exception as e:
            print(e)
        side_bar_data.append(f)
    return side_bar_data


# Helper Function
def display_features(df, feature_lookup):
    """
    This function displayes feature names and descriptions (from feature_lookup).

    Inputs:
        - df (pandas.DataFrame): The input DataFrame to be whose features to be displayed.
        - feature_lookup (dict): A dictionary containing the descriptions for the features.
    Outputs: None
    """
    numeric_columns = list(df.select_dtypes(include='number').columns)
    #for idx, col in enumerate(df.columns):
    for idx, col in enumerate(numeric_columns):
        if col in feature_lookup:
            st.markdown('Feature %d - %s' % (idx, feature_lookup[col]))
        else:
            st.markdown('Feature %d - %s' % (idx, col))


# Helper Function
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    This function fetches a dataset from a URL, saves it in .tgz format, and extracts it to a specified directory path.

    Inputs:
    - housing_url (str): The URL of the dataset to be fetched.
    - housing_path (str): The path to the directory where the extracted dataset should be saved.

    Outputs: None
    """
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

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
col1, col2 = st.columns(2)
with(col1):
    data = st.file_uploader("Upload Your Dataset", type=['csv','txt'])
# with(col2): #upload from cloud
with(col2):
    data_path = st.text_input("Enter data url", "", key="data_url")
    if(data_path):
        fetch_housing_data()
        data = os.path.join(HOUSING_PATH, "housing.csv")
        st.write("You Entered: ", data_path)
if data:
    ###################### EXPLORE DATASET #######################
    st.markdown('### Explore Dataset Features')
    df = pd.read_csv(data)
    st.write(df)
    # Restore dataset if already in memory
    st.session_state['house_df'] = df
    # Display feature names and descriptions (from feature_lookup)
    display_features(df, feature_lookup)

    # Display dataframe as table
    st.dataframe(df)

    ###################### VISUALIZE DATASET #######################
    st.markdown('### Visualize Features')
    
    # Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    # Collect user plot selection using user_input_features()
    chart_type = st.sidebar.selectbox(
        'Select Chart Type',
        options = ['Scatterplot','Histogram','Lineplot','Boxplot']
    )
    if(chart_type == 'Scatterplot'):
        x = st.sidebar.selectbox(
            'Select x-axis feature',
            options = df.columns
        )
        y = st.sidebar.selectbox(
            'Select y-axis feature',
            options = df.columns
        )
    elif(chart_type == 'Histogram'):
        x = st.sidebar.selectbox(
            'Select x-axis feature',
            options = df.columns
        )
        y = None
    elif(chart_type == 'Lineplot'):
        x = st.sidebar.selectbox(
            'Select x-axis feature',
            options = df.columns
        )
        y = None
    elif(chart_type == 'Boxplot'):
        x = None
        y = st.sidebar.selectbox(
            'Select y-axis feature',
            options = df.columns
        )
    # Collect user input features from sidebar
    side_bar_data = user_input_features(df, chart_type, x, y)

   
    # Draw plots including Scatterplots, Histogram, Lineplots, Boxplots
    #draw plots including Scatterplots, Histogram, Lineplots, Boxplots and take input features as x and y
    if(chart_type == 'Scatterplot'):
        fig = px.scatter(df, x=x, y=y)
        st.plotly_chart(fig)
    elif(chart_type == 'Histogram'):
        fig = px.histogram(df, x=x)
        st.plotly_chart(fig)
    elif(chart_type == 'Lineplot'):
        fig = px.line(df, x=x, y=y)
        st.plotly_chart(fig)
    elif(chart_type == 'Boxplot'):
        fig = px.box(df, y=y)
        st.plotly_chart(fig)
        


    
    ###################### CORRELATION ANALYSIS #######################
    st.markdown("### Looking for Correlations")
    # Collect features for correlation analysis using multiselect
    select_feature = st.multiselect(
        'Select two or features to compute correlation',
        options = df.columns,
        default = ['latitude','total_rooms']
    ) 
    # Compute correlation between selected features
    if select_feature:
        correlation, sum_statement = compute_correlation(df,select_feature)
        #st.write(correlation)
        for i in sum_statement:
            st.markdown(i)
        #st.write(sum_statement)
    # Display correlation of all feature pairs with description of magnitude and direction of correlation
    if(select_feature):
        try:
            fig =scatter_matrix(df[select_feature],figsize=(12,8))
            st.pyplot(fig[0][0].get_figure())
        except Exception as e:
            print(e)
    ###################### DOWNLOAD DATASET #######################
    st.markdown('### Download the dataset')

    csv = convert_df(df)

    st.write('Saving dataset to paml_dataset.csv')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='paml_dataset.csv',
        mime='text/csv',        
    )

    st.markdown('#### Continue to Preprocess Data')