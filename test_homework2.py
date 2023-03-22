from pages import A_Explore_Dataset, B_Preprocess_Data, C_Train_Model, D_Test_Model, E_Deploy_App
import pandas as pd
import numpy as np
import streamlit as st
import math

############## Assignment 2 Inputs #########
student_filepath = "datasets/housing/housing_paml_hw2.csv"
grader_filepath = "test_dataframe_file/housing_paml_hw2.csv"
student_dataframe = pd.read_csv(student_filepath)
grader_dataframe = pd.read_csv(grader_filepath)
e_dataframe = pd.read_csv(grader_filepath)
test_metrics = ['mean_absolute_error', 'root_mean_squared_error', 'r2_score']

# Checkpoint 1
def test_compute_corr():
    e_corr = np.array([[1,  -0.035676], [-0.035676, 1]])
    test_corr, test_str = A_Explore_Dataset.compute_correlation(
        e_dataframe, ['latitude', 'total_rooms'])
    assert np.allclose(e_corr, test_corr)
    assert test_str == [
        '- Features latitude and total_rooms are weakly negatively correlated: -0.04']

# Checkpoint 2 
def test_summarize_missing_data():
    """
    1) Number of categories with missing values.
    2) Average number of missing values per category.
    3) Number of categories with invalid value types: Average number of invalid
    value types per category.
    4) Top n Categories with most missing values.
    """
    expected_num_categories = 17.00
    expected_average_per_category = 747.24
    expected_total_missing_values =  12703.00
    expected_top_missing_categories = np.array(
        ['city', 'postcode', 'road'], dtype=object)
    student_summarize_missing_data = B_Preprocess_Data.summarize_missing_data(
        student_dataframe, 3)
    assert student_summarize_missing_data['num_categories'] == expected_num_categories
    assert math.isclose(student_summarize_missing_data['average_per_category'], expected_average_per_category, rel_tol=1e-2)
    assert student_summarize_missing_data['total_missing_values'] == expected_total_missing_values
    assert np.array_equal(student_summarize_missing_data['top_missing_categories'], expected_top_missing_categories)

# Checkpoint 3 
def test_remove_nans():
    student_remove_Nan_df = B_Preprocess_Data.remove_nans(grader_dataframe)
    assert student_remove_Nan_df.isna().sum().sum() == 0

# checkpoint 4 
def test_remove_outliers():
    """
    longitude=-127.42499999999998, -112.345
    latitude =28.259999999999998, 43.38
    Outliers for feature housing_median_age are lower than -10.5 and higher than 65.5
    Outliers for feature total_rooms are lower than -1092.0 and higher than 5692.0
    Outliers for feature total_bedrooms are lower than -230.5 and higher than 1173.5
    Outliers for feature total_bedrooms are lower than -230.5 and higher than 1173.5
    Outliers for feature households are lower than -206.5 and higher than 1093.5
    Outliers for feature median_income are lower than -0.7032249999999993 and higher than 8.012574999999998
    Outliers for feature median_house_value are lower than -97800.0 and higher than 482200.0
    """
    _, student_remove_outliers_lb_lon, student_remove_outliers_ub_lon = B_Preprocess_Data.remove_outliers(
        student_dataframe, 'longitude')
    _, student_remove_outliers_lb_lat, student_remove_outliers_ub_lat = B_Preprocess_Data.remove_outliers(
        student_dataframe, 'latitude')
    _, student_remove_outliers_lb_tb, student_remove_outliers_ub_tb = B_Preprocess_Data.remove_outliers(
        student_dataframe, 'total_bedrooms')
    _, student_remove_outliers_lb_hhold, student_remove_outliers_ub_hhold = B_Preprocess_Data.remove_outliers(
        student_dataframe, 'households')
    assert math.isclose(student_remove_outliers_lb_lon, -126.62)
    assert math.isclose(student_remove_outliers_ub_lon, -112.78)
    assert math.isclose(student_remove_outliers_lb_lat, 28.66, rel_tol=1e-2)
    assert math.isclose(student_remove_outliers_ub_lat, 42.54,rel_tol=1e-2)
    assert math.isclose(student_remove_outliers_lb_tb ,-239.5, rel_tol=1e-2)
    assert math.isclose(student_remove_outliers_ub_tb, 1196.5, rel_tol=1e-2)
    assert math.isclose(student_remove_outliers_lb_hhold ,-201.0,rel_tol=1e-2)
    assert math.isclose(student_remove_outliers_ub_hhold , 1111.0,rel_tol=1e-2)

# Checkpoint 5 
def test_one_hot_encode_feature():
    """
    Test one_hot_encode_feature(df,feature): 
    Checkpoint 5 (Page B) - Handling Text/Categorical Data. Test case: use a data structure
    """
    student_one_hot_encode_feature = B_Preprocess_Data.one_hot_encode_feature(
        student_dataframe, 'ocean_proximity')
    assert 'ocean_proximity_<1H OCEAN' in student_one_hot_encode_feature.columns
    assert 'ocean_proximity_INLAND' in student_one_hot_encode_feature.columns
    assert 'ocean_proximity_ISLAND' in student_one_hot_encode_feature.columns
    assert 'ocean_proximity_NEAR BAY' in student_one_hot_encode_feature.columns
    assert 'ocean_proximity_NEAR OCEAN' in student_one_hot_encode_feature.columns

def test_integer_encode_feature():
    student_integer_encode_feature = B_Preprocess_Data.integer_encode_feature(
        student_dataframe, 'ocean_proximity')
    assert student_integer_encode_feature['ocean_proximity'].sum() == 24009.0

# Checkpoint 6 
def test_create_feature():
    student_create_feature_df = B_Preprocess_Data.create_feature(
        student_dataframe, 'add', ['longitude', 'latitude'], 'hello')
    assert len(student_create_feature_df.columns) > len(
        student_dataframe.columns)

# CheckPoint 7 
def test_split_dataset():
    e_X = e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]
    e_Y = e_dataframe.loc[:, e_dataframe.columns.isin(['median_house_value'])]
    s_split_train_x, s_split_train_y, s_split_test_x, s_split_test_y = C_Train_Model.split_dataset(
        e_X, e_Y, 30)
    assert s_split_train_x.shape == (14448,16)

# Checkpoint 8 
def test_inspect_coefficient():
    student_dataframe_nan = B_Preprocess_Data.remove_nans(student_dataframe)

    X = student_dataframe_nan.loc[:, student_dataframe_nan.columns.isin([
                                                                        'longitude'])]
    Y = student_dataframe_nan.loc[:, student_dataframe_nan.columns.isin(
        ['median_house_value'])]
    X_train, X_val, y_train, y_val = C_Train_Model.split_dataset(X, Y, 30)
    student_Multiple_Linear_Regression = C_Train_Model.train_multiple_regression(
        X_train, y_train, 'Multiple Linear Regression')
    student_Polynomial_Regression = C_Train_Model.train_polynomial_regression(
        X_train, y_train, 'Polynomial Regression', 3, False)
    ridge_params = {"solver": ["auto", "svd"], "alpha": [1, 0.5]}
    lasso_params = {"tol": [0.001, 0.0001], "alpha": [1, 0.5]}
    student_Ridge_Regression = C_Train_Model.train_ridge_regression(
        X_train, y_train, 'Ridge Regression', ridge_params=ridge_params, ridge_cv_fold=3)
    student_Lasso_Regression = C_Train_Model.train_lasso_regression(
        X_train, y_train, 'Lasso Regression', lasso_params=lasso_params, lasso_cv_fold=3)
    expected_poly_coefficient = np.array(
        [27815.07044908, -19450.82482728, -19777.94476361])
    expected_ridge_coefficient = np.array([[2569.7090167]])
    expected_lasso_coefficient = np.array([2569.05265257])
    regression_methods_options = ['Multiple Linear Regression',
                                  'Polynomial Regression', 'Ridge Regression', 'Lasso Regression']
    models = {'Multiple Linear Regression': student_Multiple_Linear_Regression,
              'Polynomial Regression': student_Polynomial_Regression,
              'Ridge Regression': student_Ridge_Regression,
              'Lasso Regression': student_Lasso_Regression}

    student_results = C_Train_Model.inspect_coefficients(
        models, regression_methods_options
    )

    student_Multiple_Linear_Regression_coeff = np.array(
        student_results['Multiple Linear Regression'])
    student_Polynomial_Regression_coeff = np.array(
        student_results['Polynomial Regression'])
    student_Ridge_Regression_coeff = np.array(
        student_results['Ridge Regression'])
    student_Lasso_Regression_coeff = np.array(
        student_results['Lasso Regression'])
    assert np.allclose(student_Multiple_Linear_Regression_coeff,
                       np.array([1369.35463082]))
    assert np.allclose(
        student_Polynomial_Regression[-1] .coef_, expected_poly_coefficient)
    assert np.allclose(
        student_Ridge_Regression['ridgeCV'].best_estimator_.coef_, expected_ridge_coefficient)
    assert np.allclose(
        student_Lasso_Regression['lassoCV'].best_estimator_.coef_, expected_lasso_coefficient)

def flatten_dict(dictionary, key):
    return np.array([val for val in dictionary[key].values()])

# Checkpoint 9 
def test_model_metrics():
    expected_results = {
        'Multiple Linear Regression': {
            'mean_absolute_error': 84214.69579043229,
            'root_mean_squared_error': 107405.43533055007,
            'r2_score': 0.000707935606354182
        },
        'Polynomial Regression': {
            'mean_absolute_error': 82308.29766716881,
            'root_mean_squared_error': 105603.77140404365,
            'r2_score': 0.03395184402618556
        },
        'Ridge Regression': {
            'mean_absolute_error': 84214.68485060059,
            'root_mean_squared_error': 107405.43533088331,
            'r2_score': 0.0007079356001534753
        },
        'Lasso Regression': {
            'mean_absolute_error': 84214.65490126661,
            'root_mean_squared_error': 107405.43533520534,
            'r2_score': 0.0007079355197296966
        }
    }

    expected_multi = flatten_dict(
        expected_results, 'Multiple Linear Regression')
    expected_poly = flatten_dict(expected_results, 'Polynomial Regression')
    expected_ridge = flatten_dict(expected_results, 'Ridge Regression')
    expected_lasso = flatten_dict(expected_results, 'Lasso Regression')

    student_dataframe_nan = B_Preprocess_Data.remove_nans(student_dataframe)

    X = student_dataframe_nan.loc[:, student_dataframe_nan.columns.isin([
                                                                        'longitude'])]
    Y = student_dataframe_nan.loc[:, student_dataframe_nan.columns.isin(
        ['median_house_value'])]
    student_multi_reg = C_Train_Model.train_multiple_regression(
        X, Y, 'Multiple Linear Regression')
    student_poly_reg = C_Train_Model.train_polynomial_regression(
        X, Y, 'Polynomial Regression', 3, True
    )
    student_ridge_reg = C_Train_Model.train_ridge_regression(
        X, Y, 'Ridge Regression',
        {"solver": ["auto", "svd"], "alpha": [1, 0.5]}, 3
    )
    student_lasso_reg = C_Train_Model.train_lasso_regression(
        X, Y, 'Lasso Regression',
        {"tol": [0.001, 0.0001], "alpha": [1, 0.5]}, 3
    )

    models = [student_multi_reg, student_poly_reg,
              student_ridge_reg, student_lasso_reg]

    regression_methods_options = ['Multiple Linear Regression',
                                  'Polynomial Regression', 'Ridge Regression', 'Lasso Regression']

    student_results = {}
    for idx, model in enumerate(models):
        student_results[regression_methods_options[idx]] = D_Test_Model.compute_eval_metrics(
            X, Y, model, test_metrics
        )

    print(student_results)

    student_multi = flatten_dict(
        student_results, 'Multiple Linear Regression')
    student_poly = flatten_dict(student_results, 'Polynomial Regression')
    student_ridge = flatten_dict(student_results, 'Ridge Regression')
    student_lasso = flatten_dict(student_results, 'Lasso Regression')

    assert np.allclose(expected_multi, student_multi)
    assert np.allclose(expected_poly, student_poly)
    assert np.allclose(expected_ridge, student_ridge)
    assert np.allclose(expected_lasso, student_lasso)

# Checkpoint 10 
def test_plot_learning_curve():
    student_dataframe_nan = B_Preprocess_Data.remove_nans(student_dataframe)

    X = student_dataframe_nan.loc[:, student_dataframe_nan.columns.isin([
                                                                        'longitude'])]
    Y = student_dataframe_nan.loc[:, student_dataframe_nan.columns.isin(
        ['median_house_value'])]

    X_train, X_val, y_train, y_val = C_Train_Model.split_dataset(X, Y, 30)

    student_Multiple_Linear_Regression = C_Train_Model.train_multiple_regression(
        X_train, y_train, 'Multiple Linear Regression')
    student_Polynomial_Regression = C_Train_Model.train_polynomial_regression(
        X_train, y_train, 'Polynomial Regression', 3, False)
    ridge_params = {"solver": ["auto", "svd"], "alpha": [1, 0.5]}
    lasso_params = {"tol": [0.001, 0.0001], "alpha": [1, 0.5]}
    student_Ridge_Regression = C_Train_Model.train_ridge_regression(
        X_train, y_train, 'Ridge Regression', ridge_params=ridge_params, ridge_cv_fold=3)
    student_Lasso_Regression = C_Train_Model.train_lasso_regression(
        X_train, y_train, 'Lasso Regression', lasso_params=lasso_params, lasso_cv_fold=3)

    regression_methods_options = ['Multiple Linear Regression',
                                  'Polynomial Regression', 'Ridge Regression', 'Lasso Regression']
    models = [student_Multiple_Linear_Regression, student_Polynomial_Regression,
              student_Ridge_Regression, student_Lasso_Regression]

    _, student_plot_learning_curve_mul = D_Test_Model.plot_learning_curve(
        X_train, X_val, y_train, y_val, models[0], test_metrics, regression_methods_options[0])
    _, student_plot_learning_curve_pol = D_Test_Model.plot_learning_curve(
        X_train, X_val, y_train, y_val, models[1], test_metrics, regression_methods_options[1])
    _, student_plot_learning_curve_rid = D_Test_Model.plot_learning_curve(
        X_train, X_val, y_train, y_val, models[2], test_metrics, regression_methods_options[2])
    _, student_plot_learning_curve_las = D_Test_Model.plot_learning_curve(
        X_train, X_val, y_train, y_val, models[3], test_metrics, regression_methods_options[3])
    expected_mul = pd.read_csv(
        "./test_dataframe_file/Multiple Linear Regression Errors.csv")
    expected_pol = pd.read_csv(
        "./test_dataframe_file/Polynomial Regression Errors.csv")
    expected_rid = pd.read_csv(
        "./test_dataframe_file/Ridge Regression Errors.csv")
    expected_las = pd.read_csv(
        "./test_dataframe_file/Lasso Regression Errors.csv")
    pd.testing.assert_frame_equal(
        student_plot_learning_curve_mul, expected_mul)
    pd.testing.assert_frame_equal(
        student_plot_learning_curve_pol, expected_pol)
    pd.testing.assert_frame_equal(
        student_plot_learning_curve_rid, expected_rid)
    pd.testing.assert_frame_equal(
        student_plot_learning_curve_las, expected_las)