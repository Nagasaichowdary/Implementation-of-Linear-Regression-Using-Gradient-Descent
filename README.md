# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.
```
Program to implement the linear regression using gradient descent.
Developed by: Aditya Naga Sai P
RegisterNumber:212223110036
```
## Program:
```
/*
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")  
*/
```

## Output:

## data information:
<img width="558" height="222" alt="image" src="https://github.com/user-attachments/assets/5c58d1cb-0dd5-4fed-a3ad-ddfa21510707" />

## Value of x:
<img width="225" height="713" alt="image" src="https://github.com/user-attachments/assets/fed4806d-6ded-43c8-902c-ae95e03dd0ed" />

## Value of X1_scaled:
<img width="343" height="707" alt="image" src="https://github.com/user-attachments/assets/c2ced069-5291-4581-9a3b-5955922efc22" />

## predicted value:
<img width="247" height="46" alt="image" src="https://github.com/user-attachments/assets/57bf794f-6164-46e2-b615-6a36b14c3ffd" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
