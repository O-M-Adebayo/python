##### OLUWASEGUN MICHEAL ADEBAYO ######


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# importing the data

desired_width = 500
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 300)

file_name = r'C:/projectdata.dta'
df = pd.read_stata(file_name)
print(df.__len__())
print()
#print(df['ca_timechcare'].head())
print()
print('the maximum value in wellbeing is: ', df['ca_scghq1_dv'].max())
print('the minimum value in wellbeing is: ', df['ca_scghq1_dv'].min())
print()

#Data Wrangling
print(df.dtypes)
print()
print(df.describe())

#slicing the data to the needed variables: Subjective wellbeing (psychological distress) and time spent on child care
needed_variables = df[['ca_scghq1_dv', 'ca_timechcare', 'ca_hhcompb', 'ca_sex', 'ca_age', 'ca_couple']]
needed_variables.to_csv(r'C:/needed_data.csv', index=False)
data = pd.read_csv(r'C:/needed_data.csv')
print(data)
print()
#print(data['ca_timechcare'].value_counts())
print()
#
print(data.dtypes)
# print()
# #print(data.describe(include='all'))
# print()
#
###################################################################################################################
# subjective wellbeing column wrangling
data['ca_scghq1_dv'] = pd.to_numeric(data['ca_scghq1_dv'], errors='coerce').astype('Int64')
ca_scghq1_dv_mode = data['ca_scghq1_dv'].mode()
data['ca_scghq1_dv'].replace({np.nan:ca_scghq1_dv_mode}, inplace=True)
data['ca_scghq1_dv'] = data['ca_scghq1_dv'].round(0)
print(data)

data['ca_timechcare'] = pd.to_numeric(data['ca_timechcare'], errors='coerce').astype('Int64')
ca_timechcare_mode = data['ca_timechcare'].mode()
data['ca_timechcare'].replace({np.nan:ca_timechcare_mode}, inplace=True)
data['ca_timechcare'] = data['ca_timechcare'].round(0)
print(data)
####################################################################################################################


######################################################################################################################
data['ca_scghq1_dv'] = pd.to_numeric(data['ca_scghq1_dv'], errors='coerce').astype('Int64')
data['ca_timechcare'] = pd.to_numeric(data['ca_timechcare'], errors='coerce').astype('Int64')
data.dropna(subset=['ca_scghq1_dv'], axis=0, inplace=True)
data.dropna(subset=['ca_timechcare'], axis=0, inplace=True)
data = data.drop([0,1], axis=0)
print(data)
print()
#print(data[['ca_timechcare', 'ca_scghq1_dv']].describe())
######################################################################################################################


####################################################################################################################
#binning the timechcare to four levels
bins = np.linspace(min(data['ca_timechcare']), max(data['ca_timechcare']), 5)
group_names = ['V-Low', 'Low', 'Medium', 'High']
data['ca_timechcare_binned'] = pd.cut(data['ca_timechcare'], bins, labels=group_names, include_lowest=True)
print(data.head())
plt.hist(data['ca_timechcare_binned'])
sns.histplot(data['ca_timechcare_binned']).set(title = 'Four categories of time spent on child care')
plt.savefig(r'C:\'Binned_Time_Shared.png')
plt.title('Four categories of time spent on child care')
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.grid()
plt.savefig(r'C:\'Binned_Time_Shared_2.png')
plt.show()
#
print()
data_dummies = pd.get_dummies(data, prefix='Time', columns=['ca_timechcare_binned'])
data_dummies = pd.get_dummies(data_dummies, prefix='Couple', columns=['ca_couple'])
#normalizing the wellbeing data
max_norm = np.max(data_dummies['ca_scghq1_dv'])
min_norm = np.min(data_dummies['ca_scghq1_dv'])
data_dummies['ca_scghq1_dv_norm'] = ((data_dummies['ca_scghq1_dv'] - min_norm) / (max_norm - min_norm))
data_dummies['ca_scghq1_dv_norm'] = data_dummies['ca_scghq1_dv_norm'].round(2)

max_norma = np.max(data_dummies['ca_age'])
min_norma = np.min(data_dummies['ca_age'])
data_dummies['ca_age_norm'] = ((data_dummies['ca_age'] - min_norma) / (max_norma - min_norma))
data_dummies['ca_age_norm'] = data_dummies['ca_age_norm'].round(2)
sns.pairplot(data_dummies[['ca_scghq1_dv', 'ca_timechcare', 'ca_age']])
#plt.suptitle('Pairplot of Subjective Wellbeing, Time on Child Care, and Age')
plt.savefig(r'C:\'Pairplot_SubWell_Time_Age.png')
plt.show()
the_correlation = data_dummies.corr()
print('correlation: ', the_correlation)
print()
print(data_dummies)
# print()

#descriptive statistics
#means
print('Wellbeing descriptive statistics:')
print('Wellbeing_mean: ', data_dummies['ca_scghq1_dv'].mean().round(2))
print('Wellbeing_std: ', data_dummies['ca_scghq1_dv'].std().round(2))
print('Wellbeing_min: ', data_dummies['ca_scghq1_dv'].min().round(2))
print('Wellbeing_max: ', data_dummies['ca_scghq1_dv'].max().round(2))
print('Time on child care descriptive statistics:')
print('Time_care_mean: ', data_dummies['ca_timechcare'].mean().round(2))
print('Time_care_std: ', data_dummies['ca_timechcare'].std().round(2))
print('Time_care_min: ', data_dummies['ca_timechcare'].min().round(2))
print('Time_care_max: ', data_dummies['ca_timechcare'].max().round(2))
print('Age descriptive statistics:')
print('Age_mean: ', data_dummies['ca_age'].mean().round(2))
print('Age_std: ', data_dummies['ca_age'].std().round(2))
print('Age_min: ', data_dummies['ca_age'].min().round(2))
print('Age_max: ', data_dummies['ca_age'].max().round(2))
print()
print('Value counts on the categorical variables:')
print('Sex value counts: ', data['ca_sex'].value_counts())
print()
print('Couple value counts: ', data['ca_couple'].value_counts())
#####################################################################################################################

###############################################Visualizations########################################################
#sns.boxplot(data['ca_scghq1_dv'])
sns.boxplot(x = data['ca_timechcare_binned'], y = data['ca_scghq1_dv']).set(title = 'Boxplot of the Binned Time on Child Care vs Subjective Wellbeing')
plt.savefig(r'C:\'Boxplot_Time_SubWell.png')
plt.show()

plt.scatter(data['ca_timechcare'], data['ca_age'])
plt.xlabel('Time_Child_Care')
plt.ylabel('Age')
plt.title('Scatterplot of Time on Child Care vs Age')
plt.grid()
plt.savefig(r'C:\'Scatterplot_Time_Age.png')
plt.show()

plt.scatter(data['ca_timechcare'], data['ca_scghq1_dv'], color = 'r')
plt.xlabel('Time_Child_Care')
plt.ylabel('Subjective Wellbeing')
plt.title('Scatterplot of Time on Child Care vs Subjective Wellbeing')
plt.grid()
plt.savefig(r'C:\'Scatterplot_Time_GHQ.png')
plt.show()
###############################################Visualizations#######################################################


####################################################################################################################
#Multiple Linear Regression Modelling
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
X = data_dummies[['Time_V-Low', 'Time_Low', 'Time_Medium', 'Time_High', 'Couple_No', 'Couple_Yes', 'ca_age']]
y = data_dummies['ca_scghq1_dv']

#splitting the data into training and testing parts
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#training the model
lm.fit(x_train, y_train)
Yhat = lm.predict(x_test)
print("The intercept of the model is: ", lm.intercept_, "and the coefficients are: ", lm.coef_)
#print(Yhat)
print()
###############################################################################################################


#############################################
#importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
score_value = r2_score(y_test, Yhat)
print('r2_score value is: ', score_value)
#############################################

# score_value = lm.score(y_test, Yhat)
# print(score_value)
# plt.plot(Yhat/(10**10)
# plt.show()

##Visualizing the actual values and the fitted values
# ax1 = sns.distplot(data_dummies['ca_scghq1_dv'], hist=False, color='r', label='Actual Values')
sns.kdeplot(data_dummies['ca_scghq1_dv'], color='b', label='Actual values')
#sns.distplot(Yhat, hist=False, color='b', label='Fitted Values', ax=ax1)
sns.kdeplot(Yhat, color='r', label='Fitted values')
plt.yscale('log')
plt.title('Distribution plot of the actual values and fitted values')
plt.xlabel('Subjective wellbeing')
# plt.ylabel('The target')
plt.legend()
plt.grid()
plt.show()
#############################################################################################################


