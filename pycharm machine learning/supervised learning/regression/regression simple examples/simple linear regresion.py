import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''
data prepprocessing
'''
salaryset = pd.read_csv('test.csv')
x = salaryset.iloc[:,:-1].values
y = salaryset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
'''
training the simple linear regression model on the training set
'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
'''
prdicting the test set result
'''
y_pred = regressor.predict(x_test)

'''
visualising the traing set results
'''
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')##we didint call a varibale like y_pred we just insirted
plt.title('salary vs experience(training set)')
plt.xlabel('years of exprience')
plt.ylabel('salary')
plt.legend()
plt.show()
'''
visualising the test set result
'''
#print(regressor.predict([[12]])) ## to predict a value and show it
plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue') ##we can let x_train because they reult from the same equation
plt.title('salary vs experience (test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()
plt.legend()
###print(regressor.coef_)
###print(regressor.intercept_) to prin the equation of the droite finale
