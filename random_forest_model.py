# In this code, the load_data function loads a CSV file into a Pandas DataFrame. 
# The display_summary function displays summary statistics of the DataFrame, 
# such as count, mean, standard deviation, minimum, and maximum values. 
# The display_missing_values function shows the count of missing values in 
# each column of the DataFrame. The visualize_distribution function visualizes 
# the distribution of a numerical column using a histogram. 
# The visualize_relationship function shows the relationship between 
# two numerical columns using a scatter plot.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data(file_path):
    """
    Load data from a CSV file into a Pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def display_summary(df):
    """
    Display summary statistics of the DataFrame.
    """
    print(df.describe())

def display_missing_values(df):
    """
    Display the count of missing values in each column of the DataFrame.
    """
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

def visualize_distribution(df, column):
    """
    Visualize the distribution of a numerical column using a histogram.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

def visualize_relationship(df, x, y):
    """
    Visualize the relationship between two numerical columns using a scatter plot.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f'{x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

def perform_linear_regression(df, target_column, feature_columns):
    """
    Perform linear regression on the DataFrame using the specified target and feature columns.
    """
    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f'Linear Regression - Mean Squared Error: {mse}')

def perform_random_forest_regression(df, target_column, feature_columns):
    """
    Perform random forest regression on the DataFrame using the specified target and feature columns.
    """
    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f'Random Forest Regression - Mean Squared Error: {mse}')

# Example usage:
data = load_data('data.csv')
display_summary(data)
display_missing_values(data)
visualize_distribution(data, 'Age')
visualize_relationship(data, 'Height', 'Weight')

target_column = 'Target'
feature_columns = ['Feature1', 'Feature2', 'Feature3']
perform_linear_regression(data, target_column, feature_columns)
perform_random_forest_regression(data, target_column, feature_columns)
