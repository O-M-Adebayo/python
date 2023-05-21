##### OLUWASEGUN MICHEAL ADEBAYO ######


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def import_data(file_path):
    desired_width = 500
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns', 300)

    df = pd.read_stata(file_path)
    return df


def print_length(df):
    print("Length of the DataFrame: ", len(df))
    print()


def print_max_min_wellbeing(df):
    max_wellbeing = df['ca_scghq1_dv'].max()
    min_wellbeing = df['ca_scghq1_dv'].min()
    print("The maximum value in wellbeing is: ", max_wellbeing)
    print("The minimum value in wellbeing is: ", min_wellbeing)
    print()


def data_wrangling(df):
    print("Data types:")
    print(df.dtypes)
    print()

    print("Descriptive statistics:")
    print(df.describe())
    print()

    return df


def slice_data(df):
    needed_variables = df[['ca_scghq1_dv', 'ca_timechcare', 'ca_hhcompb', 'ca_sex', 'ca_age', 'ca_couple']]
    needed_variables.to_csv("needed_data.csv", index=False)
    data = pd.read_csv("needed_data.csv")
    print(data)
    print()

    print("Data types:")
    print(data.dtypes)
    print()

    return data


def wrangle_subjective_wellbeing(data):
    data['ca_scghq1_dv'] = pd.to_numeric(data['ca_scghq1_dv'], errors='coerce').astype('Int64')
    ca_scghq1_dv_mode = data['ca_scghq1_dv'].mode()
    data['ca_scghq1_dv'].replace({np.nan: ca_scghq1_dv_mode}, inplace=True)
    data['ca_scghq1_dv'] = data['ca_scghq1_dv'].round(0)
    print(data)

    return data


def wrangle_time_chcare(data):
    data['ca_timechcare'] = pd.to_numeric(data['ca_timechcare'], errors='coerce').astype('Int64')
    ca_timechcare_mode = data['ca_timechcare'].mode()
    data['ca_timechcare'].replace({np.nan: ca_timechcare_mode}, inplace=True)
    data['ca_timechcare'] = data['ca_timechcare'].round(0)
    print(data)

    return data

def drop_missing_values(data):
    data.dropna(subset=['ca_scghq1_dv'], axis=0, inplace=True)
    data.dropna(subset=['ca_timechcare'], axis=0, inplace=True)
    data = data.drop([0, 1], axis=0)
    print(data)
    print()

    return data

def bin_time_care(data):
    bins = np.linspace(min(data), max(data), 5)
    group_names = ['V-Low', 'Low', 'Medium', 'High']
    return pd.cut(data, bins, labels=group_names, include_lowest=True)

def visualize_binned_time(data):
    plt.hist(data)
    sns.histplot(data).set(title='Four categories of time spent on child care')
    plt.savefig(r'C:\'Binned_Time_Shared.png')
    plt.title('Four categories of time spent on child care')
    plt.xlabel('Categories')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(r'C:\'Binned_Time_Shared_2.png')
    plt.show()

def get_dummies(data):
    data_dummies = pd.get_dummies(data, prefix='Time', columns=['ca_timechcare_binned'])
    data_dummies = pd.get_dummies(data_dummies, prefix='Couple', columns=['ca_couple'])
    return data_dummies

def normalize_data(data):
    max_norm = np.max(data)
    min_norm = np.min(data)
    normalized_data = ((data - min_norm) / (max_norm - min_norm)).round(2)
    return normalized_data

def visualize_pairplot(data):
    sns.pairplot(data[['ca_scghq1_dv', 'ca_timechcare', 'ca_age']])
    plt.savefig(r'C:\'Pairplot_SubWell_Time_Age.png')
    plt.show()

def calculate_correlation(data):
    correlation = data.corr()
    return correlation

def descriptive_statistics(data):
    print('Wellbeing descriptive statistics:')
    print('Wellbeing_mean:', data['ca_scghq1_dv'].mean().round(2))
    print('Wellbeing_std:', data['ca_scghq1_dv'].std().round(2))
    print('Wellbeing_min:', data['ca_scghq1_dv'].min().round(2))
    print('Wellbeing_max:', data['ca_scghq1_dv'].max().round(2))
    print('Time on child care descriptive statistics:')
    print('Time_care_mean:', data['ca_timechcare'].mean().round(2))
    print('Time_care_std:', data['ca_timechcare'].std().round(2))
    print('Time_care_min:', data['ca_timechcare'].min().round(2))
    print('Time_care_max:', data['ca_timechcare'].max().round(2))
    print('Age descriptive statistics:')
    print('Age_mean:', data['ca_age'].mean().round(2))
    print('Age_std:', data['ca_age'].std().round(2))
    print('Age_min:', data['ca_age'].min().round(2))
    print('Age_max:', data['ca_age'].max().round(2))
    print()

def visualize_value_counts(data):
    print('Value counts on the categorical variables:')
    print('Sex value counts:', data['ca_sex'].value_counts())
    print()
    print('Couple value counts:', data['ca_couple'].value_counts())
    
def visualize_boxplot(data):
    sns.boxplot(x=data['ca_timechcare_binned'], y=data['ca_scghq1_dv']).set(title='Boxplot of the Binned Time on Child Care vs Subjective Wellbeing')
    plt.savefig(r'C:\'Boxplot_Time_SubWell.png')
    plt.show()

def visualize_scatterplot_time_age(data):
    plt.scatter(data['ca_timechcare'], data['ca_age'])
    plt.xlabel('Time_Child_Care')
    plt.ylabel('Age')
    plt.title('Scatterplot of Time on Child Care vs Age')
    plt.grid()
    plt.savefig(r'C:\'Scatterplot_Time_Age.png')
    plt.show()

def visualize_scatterplot_time_wellbeing(data):
    plt.scatter(data['ca_timechcare'], data['ca_scghq1_dv'], color='r')
    plt.xlabel('Time_Child_Care')
    plt.ylabel('Subjective Wellbeing')
    plt.title('Scatterplot of Time on Child Care vs Subjective Wellbeing')
    plt.grid()
    plt.savefig(r'C:\'Scatterplot_Time_GHQ.png')
    plt.show()

def perform_linear_regression(data):
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    X = data[['Time_V-Low', 'Time_Low', 'Time_Medium', 'Time_High', 'Couple_No', 'Couple_Yes', 'ca_age']]
    y = data['ca_scghq1_dv']

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lm.fit(x_train, y_train)
    Yhat = lm.predict(x_test)
    print("The intercept of the model is:", lm.intercept_, "and the coefficients are:", lm.coef_)
    print()

def calculate_r2_score(y_test, Yhat):
    from sklearn.metrics import r2_score
    score_value = r2_score(y_test, Yhat)
    print('r2_score value is:', score_value)

def visualize_distribution_plot(data, Yhat):
    sns.kdeplot(data['ca_scghq1_dv'], color='b', label='Actual values')
    sns.kdeplot(Yhat, color='r', label='Fitted values')
    plt.yscale('log')
    plt.title('Distribution plot of the actual values and fitted values')
    plt.xlabel('Subjective wellbeing')
    plt.legend()
    plt.grid()
    plt.show()

