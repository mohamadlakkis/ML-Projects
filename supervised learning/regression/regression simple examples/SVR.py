import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,  1: -1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)
# we need to rehsape y into a 2d array verticaly
y = y.reshape(len(y),1)
print(y)
'''
feature scalling ## because there is no bx and b1x its not a linear
'''
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)
print(x)
print(y)
'''
training the SVR model on the whole Dataset
'''
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

'''
predicting a new result
'''
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))
#now we need to apply reverse scalling these 2 steps are up
# we need to apply our scalling to this value here
'''
visualising the SVR results
'''
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y) , color = 'red')
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x)),color = 'blue')
plt.title('truth or bluff')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()
'''
visualising the svr with better resolutions
'''
x_grid = np.arange(min(sc_x.inverse_transform(x)),max(sc_x.inverse_transform(x)),0.1) ## here we are working with 0.1 instead of only the real value of x
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y),color='red')
plt.plot(x_grid,sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))),color='blue')
plt.title('svr regression model')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()
