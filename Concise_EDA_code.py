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

These additional data cleaning and visualization steps help to ensure the data is high
quality and informative, and prepare it for further analysis and modeling.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('data.csv')

# Print the first five rows of the data
print("First five rows of the data:")
print(data.head())

# Get the shape of the data
print("Shape of the data:")
print(data.shape)

# Check for missing values
print("Number of missing values:")
print(data.isnull().sum())

# Drop missing values
data = data.dropna()

# Get the data types of the columns
print("Data types of the columns:")
print(data.dtypes)

# Get the summary statistics of the numerical columns
print("Summary statistics of the numerical columns:")
print(data.describe())

# Get the unique values and their frequencies for categorical columns
print("Unique values and their frequencies for categorical columns:")
for column in data.select_dtypes(include=['object']):
    print(column, ":")
    print(data[column].value_counts())
    print()

# Plot histograms of the numerical columns
print("Histograms of the numerical columns:")
data.hist(bins=20, figsize=(12, 10))
plt.show()

# Plot box plots of the numerical columns
print("Box plots of the numerical columns:")
data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(12, 10))
plt.show()

# Plot correlation matrix of the numerical columns
print("Correlation matrix of the numerical columns:")
corr_matrix = data.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True)
plt.show()

# Plot scatter plots of the numerical columns against the target variable
print("Scatter plots of the numerical columns against the target variable:")
target_column = 'target'
numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
numerical_columns.remove(target_column)
sns.pairplot(data, x_vars=numerical_columns, y_vars=target_column)
plt.show()
