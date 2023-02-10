## COURSE: SPECIAL TOPICS IN THE UPPER ATMOSPHERE
## STUDENT: OLUWASEGUN MICHEAL ADEBAYO
## INPE NO: 121492/2022
## TEACHER: DR. ALEXANDRE A. PIMENTA
## DATE: SEPTEMBER 30, 2022.

# __________________________________________________________________________________________________________
# import the libraries
import pandas as pd # module for dataframe wrangling
from numpy import exp # exponential math function from numpy
import matplotlib.pyplot as plt # module for plotting
plt.style.use(['science', 'notebook', 'grid']) # graph display style

# importing the neutral density data obtained from MSIS00
df = pd.read_csv(r"C:\Special_Topics_2_Pimenta/neutral_data_May10.csv")

# preview the first-five of the data
print(df.head())

# store the alpha data in a list
alpha = [0.1, 0.3, 0.6]  # constant $\alpha$
# indexing temperature from the data
T = df['Temp_N (K)']

# rate of reactions coefficients
k1 = 1.6E-19*exp(-197.0/T)
k4 = 5.7E-34 * ((T/300)**(-2.37))
k6N2 = k4/(1.4E-10 * (exp(-470/T)))
k6O2 = k4/5.7E-34 * ((T/300)**(-2.62))
# Volume emission rate for alpha
VER0 = alpha[0] * k1 * df['O (cm-3)'] * df['O2 (cm-3)'] # alpha = 0.1
VER1 = alpha[1] * k1 * df['O (cm-3)'] * df['O2 (cm-3)'] # alpha = 0.3
VER2 = alpha[2] * k1 * df['O (cm-3)'] * df['O2 (cm-3)'] # alpha = 0.6

# plot the results
plt.plot(VER0, df['Height (km)'], label= r'$\alpha$ = 0.1')
plt.plot(VER1, df['Height (km)'], label= r'$\alpha$ = 0.3')
plt.plot(VER2, df['Height (km)'], label= r'$\alpha$ = 0.6')
plt.legend(fontsize=10)
plt.ylabel('Height (km)', fontsize=12)
plt.xlabel('Volume Emission Rate (cm$^{-3}$s$^{-1}$)Na', fontsize=12)
plt.title('Sao Jose dos Campos (23.22$^{0}$S, 45.89$^{0}$W)\nMay 10, 2013 22:00h (LT)', fontsize=10)
#plt.savefig(r'C:\Special_Topics_2_Pimenta/VER_Na.png', dpi=150, bbox_inches='tight')
plt.show()

