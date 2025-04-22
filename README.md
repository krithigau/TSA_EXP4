# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 15.04.2025

### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
## Importing Necessary Libraries and Loading the Dataset
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#Load dataset
data=pd.read_csv('AirPassengers.csv')
```
## Declare required variables and set figure size, and visualise the data
```
N=1000
plt.rcParams['figure.figsize'] = [12, 6] #plt.rcParams is a dictionary-like object in Mat
X=data['#Passengers']
plt.plot(X)
plt.title('Original Data')
plt.show()
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()
```
## Fitting the ARMA(1,1) model and deriving parameters
```
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']
```
## Simulate ARMA(1,1) Process
```
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()
```
## Plotting Simulated ARMA(2,2) Data
```
#Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()
# Fitting the ARMA(1,1) model and deriving parameters
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']
#Simulate ARMA(2,2) Process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()
#Plot ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
```
OUTPUT:

![Screenshot 2025-04-22 082821](https://github.com/user-attachments/assets/7e374217-7801-4ca7-b201-0682f8beb69c)

Autocorrelation and Partial Autocorrelation:

![Screenshot 2025-04-22 082853](https://github.com/user-attachments/assets/d57060a5-1a5d-4b74-b241-072a56dbcb0e)

SIMULATED ARMA(1,1) PROCESS:

![Screenshot 2025-04-22 082904](https://github.com/user-attachments/assets/9e4c2989-197d-47e7-9441-6fc237afec6b)

Partial Autocorrelation 

![Screenshot 2025-04-22 085712](https://github.com/user-attachments/assets/3d60d75e-ce21-4026-a4b3-984ef466c33b)

Autocorrelation

![Screenshot 2025-04-22 085702](https://github.com/user-attachments/assets/81c416c1-82fe-4079-8826-e2e1a625fd28)


SIMULATED ARMA(2,2) PROCESS:
![Screenshot 2025-04-22 083045](https://github.com/user-attachments/assets/57a0aafa-774a-4166-92f6-6695fc6ebe78)

Partial Autocorrelation

![Screenshot 2025-04-22 083027](https://github.com/user-attachments/assets/49afd750-f72d-47fe-b01b-c4ad164e5dd2)


Autocorrelation

![Screenshot 2025-04-22 083018](https://github.com/user-attachments/assets/ed62c3b3-f036-4eef-b139-9d384a8e030a)


RESULT:
Thus, a python program is created to fir ARMA Model successfully.
