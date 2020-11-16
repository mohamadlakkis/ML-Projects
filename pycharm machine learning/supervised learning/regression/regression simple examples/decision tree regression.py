# the decision tree regression model is not really well adapted to theses simple datasets
# with only one feature and one dependent variable vector
# in decision tree regression don't have to apply feature scaling
# not for 2D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,  1: -1].values
y = dataset.iloc[:, -1].values
'''
training the decision tree on the whole dataset
'''
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)
'''
predicting a new result
'''
predicted_value = regressor.predict([[6.5]])
'''
visualising the decision tree regression result(high resolution)
'''
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('decision tree  regression model')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()