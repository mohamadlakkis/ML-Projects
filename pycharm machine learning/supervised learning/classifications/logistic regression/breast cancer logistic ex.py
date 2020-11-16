import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('breast_cancer.csv')
x = dataset.iloc[:,1: -1].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import multilabel_confusion_matrix,accuracy_score
matrix = multilabel_confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test,y_pred)
print(matrix)
print(acc)
ex = classifier.predict_proba([[1,2,4,5,6,7,8,9,3]])
print(ex)