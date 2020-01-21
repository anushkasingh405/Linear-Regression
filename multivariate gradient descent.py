import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

link='./auto-mpg.csv'
data_pandas = pd.read_csv(link)

####missing value treatment
data_pandas=data_pandas.replace('?','')
data_pandas=data_pandas.fillna(data_pandas.mean(), inplace=True)
data_pandas.horsepower = pd.to_numeric(data_pandas.horsepower, errors='coerce').fillna(0).astype(np.int64)

#####data seperation
X_train= data_pandas.iloc[80:,1:8]
X_train = sc.fit_transform(X_train)
X_train = np.c_[np.ones(len(X_train),dtype='int64'),X_train]
y_train=data_pandas.iloc[80:,0]
##gradient descent
m = len(y_train)
c = 0
theta=np.zeros(X_train.shape[1])
theta=theta.transpose()
alpha = 0.01 ##putting 0.0001 was not resulting into a good accuracy
iter1 = 5000

def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J

def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
 
    for iteration in range(iterations):
 #print(iteration)
 # Hypothesis Values
        h = X.dot(B)
 # Difference b/w Hypothesis and Actual Y
        loss = h - Y
 # Gradient Calculation
        gradient = X.T.dot(loss) / m
 # Changing Values of B using Gradient
        B = B - alpha * gradient
 # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
    print("Accuracy of this model",r2_score(Y, h))
    plt.xlabel('iterations')
    plt.ylabel('cost')
    plt.plot(range(0,5000),cost_history)
    plt.show()
    return(B)
    
theta=gradient_descent(X_train,y_train,theta,alpha,iter1)

