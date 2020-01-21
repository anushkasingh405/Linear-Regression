import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

link='./auto-mpg.csv'
data_pandas = pd.read_csv(link)

##missing value treatment
data_pandas=data_pandas.replace('?','')
data_pandas=data_pandas.fillna(data_pandas.mean(), inplace=True)
data_pandas.horsepower = pd.to_numeric(data_pandas.horsepower, errors='coerce').fillna(0).astype(np.int64)

####linear regression
X=data_pandas['horsepower'].values.reshape(-1,1)
X=preprocessing.scale(X)
y=data_pandas['mpg'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)

#outliers treatment
for col in data_pandas.columns:
    data=data_pandas[[col]]
    Q1,Q3=np.percentile(data,[25,75])
    data_pandas.is_copy = False
    data_pandas[col]= np.clip(data_pandas,Q1,Q3)

####ploting map
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
plt.title("We Predict The Analysis")
print('Coefficients: ', lm.coef_)
print('intercept: ', lm.intercept_)
print('accuracy',r2_score(y_test, y_pred))

