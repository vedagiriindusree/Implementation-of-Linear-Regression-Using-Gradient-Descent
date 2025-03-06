# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:Vedagiri Indu Sree 
RegisterNumber: 212223230236 
*/
```
```
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
```
```
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
```
```
#Learn model parameters
theta = linear_regression(X1_Scaled, Y1_Scaled)
#Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_Scaled), theta)
prediction = prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(f"Predicted value: (pre)")
```

## Output:
![image](https://github.com/user-attachments/assets/0a709d39-5645-4188-b3a7-fec345a7fe86)

## Values of X and Y
![image](https://github.com/user-attachments/assets/9fe58d09-d0f6-4755-ae9b-17947d46504c)

![image](https://github.com/user-attachments/assets/59fa1388-0d37-4c5e-a5e4-2f5d10d109d3)

## Predicted Value
![image](https://github.com/user-attachments/assets/7ef530d8-587b-4eb7-8371-27a9621cb5a7)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
