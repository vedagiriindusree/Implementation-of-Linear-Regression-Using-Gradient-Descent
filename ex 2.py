#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term
    X = np.c_[np.ones(len(X1)), X1]
    # Initialize theta with zeros
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    # Perform gradient descent
    for _ in range(num_iters):
        # calculate prediction
        predictions = (X).dot(theta).reshape(-1,1)
        # calculate errors
        errors = (predictions - y).reshape(-1,1)
        # Update theta using gradient descent
        theta -= learning_rate * (1/len(X1)) * X.T.dot(errors)
    return theta
data = pd.read_csv('50_Startups.csv')
print(data.head())


# In[2]:


# Assuming the last column is your target variable 'y' and the preceding columns a
X = (data.iloc[1:, :-2].values)
print(X)
X1 = X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)


# In[3]:


#Learn model parameters
theta = linear_regression(X1_Scaled, Y1_Scaled)
#Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_Scaled), theta)
prediction = prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
print("Name:Vedagiri Indu Sree")
print("Reg no:21222323036")


# In[ ]:




