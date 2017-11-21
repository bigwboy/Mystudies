import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import math


from sklearn.linear_model import SGDRegressor


iris =datasets.load_iris()
X = iris.data[:,[2,3]]
y=iris.target
#print(np.unique(y))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
sc=StandardScaler()
sc.fit(X_train)
X_train_std= sc.transform(X_train)
X_test_std= sc.transform(X_test)



X_COMBINDED_STD=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
weights,params=[],[]

for c in np.arange(-5,5):
    lr= LogisticRegression(C=math.pow(10,c),random_state=0)
    lr.fit(X_train_std,y_train)
    weights.append(lr.coef_[1])
    params.append(math.pow(10,c))