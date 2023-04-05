# This code estimate the mesospheric temperature profile 
# for a given photon count data as an input


# --------------------------------------- BEGINNING OF CODE -----------------------------------------------
####### TEMPERATURE PROFILING FROM RAYLEIGH SCATTERING #########

# AUTHOR: OLUWASEGUN MICHEAL ADEBAYO
# INPE NO: ------
# COURSE: SPECIAL TOPICS IN UPPER ATMOSPHERE
# PROFESSORS: DR. PAULO PRADO AND DRA. VANIA ANDRIOLI

#-------------------------------------------------------------------------------------------------------
# Importing the necessary libraries
import pandas as pd # Pandas for table data analysis
import numpy as np # Numpy for array operations
import matplotlib.pyplot as plt # Matplotlib for data visualization
plt.style.use(['science', 'notebook', 'grid']) # SciencePlots for scientific plots

#-------------------------------------------------------------------------------------------------------
# Parameters
dz = 0.3E3 # change in z in m
z0 = 0.6E3 # Station altitude in m
M = 0.02897 # air molecular weight from the internet kg/mol
g_surface = 9.81 # m/s-2
R = 8.3144598 # universal gas constant in J
rad_of_earth = 6371E3 # radius of the earth in m
B = 700 # background noise
seed_pressure = 3.97 # seeding pressure in pascal

#-------------------------------------------------------------------------------------------------------
# Loading the data
lidar_data = pd.read_csv(r"C:/paulo_data.csv")
model_data = pd.read_csv(r"C:/model_temp_data.csv")
print(lidar_data.head())
# Restructuring the data due to the top-to-bottom condition of CH method
# With the inversion technique we can integrate normally (top-to-bottom)
lidar_data = lidar_data.iloc[::-1] # Inverting the data (149-0)
lidar_data = lidar_data.reset_index(drop=True) # Resetting the data index (0-149)
n = len(lidar_data) # length of the data (150)

#-------------------------------------------------------------------------------------------------------
# Creating empty lists to store the values
pressure = [] # Empty list to store pressure values
X_factor = [] # Empty list to store X factor values
temperature = [] # Empty list to store temperature values
gravity = [] # Empty list to store gravity values

#-------------------------------------------------------------------------------------------------------
# Looping through the length of the data
for i in range(n):
    P = seed_pressure # Seeding pressure for the first iteration
    for j in range(i):
        P = (lidar_data['C*(z-z0)**2'][j] * g_surface * dz) + P # Integrating the pressure
    pressure.append(P) # Store the pressure in the created empty list
    X = (lidar_data['C*(z-z0)**2'][i] * g_surface * dz) / pressure[i] # Calculate the X factor
    X_factor.append(X) # Store the X factor in the created empty list
    T = (M * g_surface * dz) / (R * np.log(1 + X)) # Calculate the temperature
    temperature.append(T) # Store the temperature in the created empty list
    pressure[i] = P # Cumulative pressure

#----------------------------------------------------------------------------------------------------------
# Data Smoothing using numpy convolution method
kernel_size = 30 # Stating the kernel size
kernel = np.ones(kernel_size) / kernel_size # Estimating kernel of the data
temp_smoothed = np.convolve(temperature, kernel, mode='same') # Applying kernel on the data for smoothing

#----------------------------------------------------------------------------------------------------------
# Data visualization
plt.plot(temperature, lidar_data['z-z0'], '.--', lw=0.7, label=' Computed Temp') # Raw temperature data with corrected altitude
plt.plot(temp_smoothed, lidar_data['z-z0'], '.--', lw=0.7, label='Smoothed Temp') # Smoothed temperature data
plt.plot(model_data['temp'], model_data['altitude'], '.--', lw=0.7, label='MSIS00 Temp')
plt.xlabel('Temperature (K)') # x label - temperature in K
plt.ylabel('Altitude (km)') # y label - altitude in km
plt.title('Temperature profile from Rayleigh Scattering\nSao Jose dos Campos: 23$^{0}$S, 46$^{0}$W\nJuly 26, 2016',
          fontsize=10) # Plot title
plt.legend(fontsize=10) # legend parameters
plt.show() # Display the plot

#----------------------------------------------- END OF CODE --------------------------------------------
