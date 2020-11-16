import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,:-1].values # in this ex we didnt import the first column beacuse it is the same of the 2
y = dataset.iloc[:,-1].values
'''
training the linear regression model on the whole dataset
'''
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x,y)
'''
training the polynominal regression model on the whole dataset
'''
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 2) # is the expsoant of x
x_poly = poly_reg.fit_transform(x) # we created a new matrix of feaurs x
linreg_2 = LinearRegression()
linreg_2.fit(x_poly,y)
'''
visualising the linear regression rsults
'''
plt.scatter(x,y,color='red')
plt.plot(x,linreg.predict(x),color='blue')
plt.title('linera regresson model')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()
'''
visualisinf the polynominal regression results
'''
plt.scatter(x,y,color='red')
plt.plot(x,linreg_2.predict(x_poly),color='blue')
plt.title('polynominal regression model')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()
'''
visualising the polynominal results for higher resolution   and smoother curve
'''
x_grid = np.arange(min(x),max(x),0.1) ## here we are working with 0.1 instead of only the real value of x
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,linreg_2.predict(poly_reg.fit_transform(x_grid)),color='blue')
plt.title('polynominal regression model')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()
'''
predicting a new result with linear regression
'''
print(linreg.predict([[6.5]]))
'''
predicting a new result with polynominal regression
'''
print(linreg_2.predict(poly_reg.fit_transform([[6.5]])))