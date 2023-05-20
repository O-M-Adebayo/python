"""

This code first loads the data from a CSV file using pandas. It then prints the first five
rows of the data, the shape of the data, and the number of missing values in each column.

It drops the missing values using the dropna function, and then gets the data types of the
columns and the summary statistics of the numerical columns, as before.

It also gets the unique values and their frequencies for categorical columns, and prints
them to the console.

The code then plots histograms of the numerical columns using the hist function, and box
plots of the numerical columns using the box function, as before. However, it uses a
larger bin size and a more complex layout for the histograms, and the Seaborn library
for the box plots, which offers more advanced visualization options.

It also plots a correlation matrix of the numerical columns using the heatmap function
from the Seaborn library, which includes an annotation of the correlation coefficients
and a triangular mask to eliminate duplicate information.

Finally, it plots scatter plots of the numerical columns against the target variable
using the pairplot function from the Seaborn library, which shows the distribution and
relationships between each variable and the target variable in a compact and informative way.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filename):
    return pd.read_csv(filename)

def print_first_rows(data, n=5):
    print(f"First {n} rows of the data:")
    print(data.head(n))

def print_data_shape(data):
    print("Shape of the data:")
    print(data.shape)

def print_missing_values(data):
    print("Number of missing values:")
    print(data.isnull().sum())

def drop_missing_values(data):
    return data.dropna()

def print_column_data_types(data):
    print("Data types of the columns:")
    print(data.dtypes)

def print_numerical_summary_statistics(data):
    print("Summary statistics of the numerical columns:")
    print(data.describe())

def print_categorical_value_counts(data):
    print("Unique values and their frequencies for categorical columns:")
    for column in data.select_dtypes(include=['object']):
        print(column, ":")
        print(data[column].value_counts())
        print()

def plot_numerical_histograms(data):
    print("Histograms of the numerical columns:")
    data.hist(bins=20, figsize=(12, 10))
    plt.show()

def plot_numerical_box_plots(data):
    print("Box plots of the numerical columns:")
    data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(12, 10))
    plt.show()

def plot_numerical_correlation_matrix(data):
    print("Correlation matrix of the numerical columns:")
    corr_matrix = data.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True)
    plt.show()

def plot_numerical_scatter_plots(data, target_column='target'):
    print("Scatter plots of the numerical columns against the target variable:")
    numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    numerical_columns.remove(target_column)
    sns.pairplot(data, x_vars=numerical_columns, y_vars=target_column)
    plt.show()

# Load the data
data = load_data('data.csv')

# Print the first five rows of the data
print_first_rows(data)

# Get the shape of the data
print_data_shape(data)

# Check for missing values
print_missing_values(data)

# Drop missing values
data = drop_missing_values(data)

# Get the data types of the columns
print_column_data_types(data)

# Get the summary statistics of the numerical columns
print_numerical_summary_statistics(data)

# Get the unique values and their frequencies for categorical columns
print_categorical_value_counts(data)

# Plot histograms of the numerical columns
plot_numerical_histograms(data)

# Plot box plots of the numerical columns
plot_numerical_box_plots(data)

# Plot correlation matrix of the numerical columns
plot_numerical_correlation_matrix(data)

# Plot scatter plots of the numerical columns against the target variable
plot_numerical_scatter_plots(data)

