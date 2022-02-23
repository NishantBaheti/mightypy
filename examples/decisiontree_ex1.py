
from mightypy.ml import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.datasets import load_iris,load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


dataset = load_iris()

X = dataset.data
y = dataset.target
feature_name = dataset.feature_names
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.10)

dt = DecisionTreeClassifier(criteria='gini')
dt.train(X=X_train,y=y_train,feature_name=feature_name)
y_pred = dt.predict_probability(X_test)
print(pd.DataFrame(np.concatenate((y_test.reshape(-1,1),y_pred),axis=1),columns=['original','predicted']))


dt = DecisionTreeClassifier(criteria='entropy')
dt.train(X=X_train,y=y_train,feature_name=feature_name)
y_pred = dt.predict_probability(X_test)
print(pd.DataFrame(np.concatenate((y_test.reshape(-1,1),y_pred),axis=1),columns=['original','predicted']))


boston_dataset = load_boston()

dataset = pd.DataFrame(data=boston_dataset['data'],columns=boston_dataset.feature_names)
dataset['target'] = boston_dataset["target"]

X = dataset[["RM","AGE","DIS","LSTAT"]].values
y = boston_dataset.target.reshape(-1,1)

feature_name = ["RM","AGE","DIS","LSTAT"]
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.10)

dt = DecisionTreeRegressor(max_depth=100,min_samples_split=3)
dt.train(X=X_train,y=y_train,feature_name=list(feature_name))
y_pred = dt.predict(X_test)
df = pd.DataFrame(np.concatenate((y_test.reshape(-1,1),y_pred),axis=1),columns=['original','predicted'])

plt.plot(df['original'])
plt.plot(df['predicted'])
plt.show()


mean_squared_error(df['original'],df['predicted'])
