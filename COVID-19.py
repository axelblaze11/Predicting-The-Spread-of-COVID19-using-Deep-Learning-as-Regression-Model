# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 19:17:45 2020

@author: Axel Blaze
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt 
import datetime
confirmed=pd.read_csv(r'F:\Project\Machine Learning & Artificial Intelligence\Project 6\Dataset\time_series_covid_19_confirmed.csv')
deaths=pd.read_csv(r'F:\Project\Machine Learning & Artificial Intelligence\Project 6\Dataset\time_series_covid_19_deaths.csv')
recovered=pd.read_csv(r'F:\Project\Machine Learning & Artificial Intelligence\Project 6\Dataset\time_series_covid_19_recovered.csv')
columns = confirmed.keys()
confirm = confirmed.iloc[:, 4:108]
death = deaths.iloc[:, 4:108]
recoveries = recovered.iloc[:, 4:108]
dates = confirm.keys()
cases = []
total_death = [] 
mortality_rate = []
total_recovered = []
for i in dates:
    confirmed_sum = confirm[i].sum()
    death_sum = death[i].sum()
    recovered_sum = recoveries[i].sum()
    cases.append(confirmed_sum)
    total_death.append(death_sum)
    mortality_rate.append(death_sum/confirmed_sum)
    total_recovered.append(recovered_sum)
   
days_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
cases = np.array(cases).reshape(-1, 1)
total_death = np.array(total_death).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)

day_in_future=15
future_predict = np.array([i for i in range(len(dates)+day_in_future)]).reshape(-1, 1)
dates_adjusted = future_predict[:-15]

start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_dates = []
for i in range(len(future_predict)):
    future_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_1_22, cases, test_size=0.15, shuffle=False)
X_train_recovered, X_test_recovered, y_train_recovered, y_test_recovered = train_test_split(days_1_22, total_recovered, test_size=0.15, shuffle=False)
X_train_death, X_test_death, y_train_death, y_test_death = train_test_split(days_1_22, total_death, test_size=0.15, shuffle=False)

y = y_train_confirmed.ravel()
y_train_confirmed = np.array(y).astype(int)
y = y_train_recovered.ravel()
y_train_recovered = np.array(y).astype(int)
y = y_train_death.ravel()
y_train_death = np.array(y).astype(int)

model= Sequential()

model.add(Dense(100, input_dim=1, kernel_initializer='normal' , activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

model.fit(X_train_confirmed, y_train_confirmed, epochs=500, batch_size=2,  verbose=1, validation_split=0.2)
y_pred_confirmed=model.predict(X_test_confirmed)
print('MAE:', mean_absolute_error(y_pred_confirmed, y_test_confirmed))
print('MSE:',mean_squared_error(y_pred_confirmed, y_test_confirmed))
print(model.predict(future_dates))
plt.plot(future_dates)

model.fit(X_train_recovered, y_train_recovered, epochs=500, batch_size=2,  verbose=1, validation_split=0.2)
y_pred_recovered=model.predict(X_test_recovered)
print('MAE:', mean_absolute_error(y_pred_recovered, y_test_recovered))
print('MSE:',mean_squared_error(y_pred_recovered, y_test_recovered))
print(model.predict(future_dates))
plt.plot(future_dates)

model.fit(X_train_death, y_train_death, epochs=500, batch_size=2,  verbose=1, validation_split=0.2)
y_pred_death=model.predict(X_test_death)
print('MAE:', mean_absolute_error(y_pred_death, y_test_death))
print('MSE:',mean_squared_error(y_pred_death, y_test_death))
print(model.predict(future_dates))
plt.plot(future_dates)

plt.plot(X_test_confirmed,y_pred_confirmed)
plt.plot(X_test_confirmed,y_test_confirmed)

plt.plot(X_test_recovered,y_pred_recovered)
plt.plot(X_test_recovered,y_test_recovered)

plt.plot(X_test_death,y_pred_death)
plt.plot(X_test_death,y_test_death)

plt.plot(cases)
plt.plot(total_recovered,color='green')
plt.plot(total_death,color='red')
