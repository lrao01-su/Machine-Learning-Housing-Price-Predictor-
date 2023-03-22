import streamlit as st
import pandas as pd                     
import numpy as np
from pages.B_Preprocess_Data import remove_nans
import random
from sklearn.preprocessing import OrdinalEncoder
random.seed(10)
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 2 - Predicting Housing Prices Using Regression")

#############################################

st.title('Deploy Application')

#############################################
enc = OrdinalEncoder()

# Checkpoint 11
def deploy_model(df):
    """
    Deploy trained regression model trained on 
    Input: 
        - df: pandas dataframe with trained regression model including
            number of bedrooms, number of bathrooms, desired city, proximity to water features
    Output: 
        - house_price: predicted house price
    """
    house_price=None
    model=None
    if('deploy_model' in st.session_state):
        model = st.session_state['deploy_model']
    house_price = model.predict(df)
    return house_price

# Helper Function
def is_valid_input(input):
    """
    Check if the input string is a valid integer or float.

    Input: 
        - input: string, char, or input from a user
    Output: 
        - True if valid input; otherwise False
    """
    try:
        num = float(input)
        return True
    except ValueError:
        return False
    
# Helper Function
def decode_integer(original_df, decode_df, feature_name):
    """
    Decode integer integer encoded feature

    Input: 
        - original_df: pandas dataframe with feature to decode
        - decode_df: dataframe with feature to decode 
        - feature: feature to decode
    Output: 
        - decode_df: Pandas dataframe with decoded feature
    """
    original_dataset[[feature_name]]= enc.fit_transform(original_dataset[[feature_name]])
    decode_df[[feature_name]]= enc.inverse_transform(st.session_state['X_train'][[feature_name]])
    return decode_df

###################### FETCH DATASET #######################


df=None
# Fill in code below
def re_dataset():
    df=None
    if 'house_df' not in st.session_state:
        data = st.file_uploader('Upload a Dataset', type=['csv', 'txt'], key='dataset_d')
        if data:
            df = pd.read_csv(data)
            st.session_state['house_df'] = df
        else:
            return None
    else:
        df = st.session_state['house_df']
    return df
df = re_dataset()


###################### Deploy App ######################

if df is not None:
    df.dropna(inplace=True)
    st.markdown('### Interested in moving to Calfornia? Predict housing prices in California based on yout preferences.')

    # Perform error checking for strings, chars, etc (garbage)

    # Input Housing Price Range
    st.markdown('## What is your desired price range?')
    min_price_range = st.number_input('Insert a minimun price')
    max_price_range = st.number_input('Insert a maximun price')
    st.write('The price range selected ${} - ${}'.format(min_price_range,max_price_range))

    # Input users input features
    user_input={}

    # Input Number of Bedrooms
    num_bedrooms = 0
    if('number_bedrooms' in st.session_state['X_train'].columns):
        st.markdown('## How many bedrooms?')
        num_bedrooms = st.slider('How many bedrooms?', 1, 10, 1)
        if(num_bedrooms):
            user_input['number_bedrooms'] = num_bedrooms
            st.write("You selected ", num_bedrooms, 'bedrooms')

    # InputNumber of Bathrooms
    num_bathrooms = 0
    if('number_bathrooms' in st.session_state['X_train'].columns):
        num_bathrooms = st.slider('How many bathrooms?', 1, 10, 1)
        if(num_bathrooms):
            user_input['number_bathrooms'] = num_bathrooms
            st.write("You selected ", num_bathrooms, 'bathrooms')

    # Input City
    original_dataset =pd.read_csv("./datasets/housing/housing_paml_hw2.csv")
    city_select=None
    if any(col.startswith('city') for col in st.session_state['X_train'].columns):

        decode_df = pd.DataFrame()
        user_options = None
        st.markdown("## Select a desired city")
        # decode integer and one-hot encded input data
        if('integer_encode' in st.session_state and st.session_state['integer_encode'].get('city')):
            decode_df = decode_integer(original_dataset,decode_df,'city')
            user_options = decode_df['city'].unique()
        elif('one_hot_encode' in st.session_state and st.session_state['one_hot_encode'].get('city')):
            city_cols = [col for col in st.session_state['X_train'].columns if col.startswith('city')]
            for col in city_cols:
                decode_df[col] = 0
                user_input[col]= 0
            user_options= decode_df.columns

        # Select desired city
        city_select = st.selectbox(
        label='Select a desired city',
        options=user_options,
        key='city_selectbox',
        index=8
        ) 
        if(city_select):
            st.write("You selected ", city_select, 'city')
            if('integer_encode' in st.session_state and st.session_state['integer_encode'].get('city')):
                user_input['city'] = enc.transform([[city_select]])[0]
            else: 
                user_input[city_select] = 1
                
    # Input Proximity to the water
    prox_water_select=None
    if any(col.startswith('ocean_proximity') for col in st.session_state['X_train'].columns):
        decode_df =pd.DataFrame()
        user_options =None
        st.markdown("## Select a desired proximity to water")
        if('integer_encode' in st.session_state and st.session_state['integer_encode'].get('ocean_proximity')):
            decode_df = decode_integer(original_dataset,decode_df,'ocean_proximity')
            user_options =decode_df['ocean_proximity'].unique()

        elif('one_hot_encode' in st.session_state and st.session_state['one_hot_encode'].get('ocean_proximity')):
            ocean_proximity_cols = [col for col in st.session_state['X_train'].columns if col.startswith('ocean_proximity')]
            for col in ocean_proximity_cols:
                decode_df[col] = 0
                user_input[col]= 0
            user_options = decode_df.columns

        prox_water_select = st.selectbox(
        label='Select a desired proximity to water',
        options=user_options,
        key='prox_water_select',
        index=min(0, len(user_options) - 1))

        if(prox_water_select):
            st.write("You selected ", prox_water_select, 'city')
            if('integer_encode' in st.session_state and st.session_state['integer_encode'].get('ocean_proximity')):
                user_input['ocean_proximity'] =enc.transform([[prox_water_select]])[0] 
            else:
                user_input[prox_water_select] = 1
    
    # Create a DataFrame from the selected features dictionary
    selected_features_df = pd.DataFrame.from_dict(user_input, orient='index').T

    # To get the mean value for unused data
    for col in st.session_state['X_train'].columns:
        if(col not in selected_features_df.columns):
            selected_features_df[col]= st.session_state['X_train'][col].mean()
    
    # Select column order of main DataFrame
    main_df_col_order = st.session_state['X_train'].columns.tolist()
    
    # Reindex the selected_features_df DataFrame with the column order of the Main DataFrame
    selected_features_df = selected_features_df.reindex(columns=main_df_col_order)
    st.write("# Predict Housing Price")

    ###################### Predict Housing Price #######################
    house_price=None
    if('deploy_model' in st.session_state and st.button('Predict Housing Price')):
        house_price = deploy_model(selected_features_df)
        if(house_price is not None):
            # Display price
            st.write('The housing price is {0:.2f}'.format(house_price[0][0]))
            if(house_price<min_price_range):
                st.write("The house price is below your budget.")
            elif(max_price_range>=house_price>=min_price_range):
                st.write("Congratulations! The house price is within your budget.")
            else:
                st.write("The house price is significantly above your budget.")

    # Show map of California
    st.markdown('## Explore California Housing Market')
    if (('lat' in df.columns or 'latitude' in df.columns) and ('lon' in df.columns or 'longitude' in df.columns)):
        df = remove_nans(df)
        st.map(df)