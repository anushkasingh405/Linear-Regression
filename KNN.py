import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

##loading data
link='./haberman.csv'
data_pandas = pd.read_csv(link)

##dividing data
x=data_pandas[['age','year','auxillary nodes']]
y=data_pandas[['status']]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=3)

##determine the best value of k
k_range=range(1,26)
scores={}
scores_list=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    ## accuracy score
    scores[k]=metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))
plt.plot(k_range,scores_list)
plt.show()
###determining best value ok k from the graph and fitting the data into the model
knn=KNeighborsClassifier(n_neighbors=12)
knn.fit(x_train,y_train)
y_predict=knn.predict(x_test)
print(metrics.accuracy_score(y_test,y_predict))
