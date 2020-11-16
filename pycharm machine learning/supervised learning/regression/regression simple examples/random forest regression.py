# its taking multiple algorithms or one multiple times and doig the average on k data
# then getting the average of all the predicted avereges
# this model is for high dimension dataset not for just one x and ne y (like decisuion tree)
# this is not for 2D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,  1: -1].values
y = dataset.iloc[:, -1].values
'''
training the random forest regression model on the whole dataset
'''
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10, random_state=0) #n_estimators how many trees
regressor.fit(x, y)
'''
predicting a new result
'''
regressor.predict([[6.5]])

'''
visualising  the random forest  regression result(high resolution)
'''
x_grid = np.arange(min(x),max(x),0.1) ## here we are working with 0.1 instead of only the real value of x
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('random forest  regression model')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()
