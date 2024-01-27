'''
####
data preprocessing
####
'''

import numpy as np
import pandas as pd
import tensorflow as tf
dataset = pd.read_excel('Folds5x2_pp.xlsx')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

'''	
#####
Building the ANN	
####	
'''

'''	
initializing the ANN	
'''

ann = tf.keras.models.Sequential()

'''	
adding the first hidden layer	
'''

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

'''	
adding the second hidden layer	
'''

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

'''	
adding the output layer	
'''

ann.add(tf.keras.layers.Dense(units=1))

'''	
####	
training the ANN	
####	
'''

'''	
compiling the ANN	
'''

ann.compile(optimizer='adam', loss='mean_squared_error')

'''	
training the ANN on the training set	
'''

ann.fit(x_train, y_train, batch_size=32, epochs=100)

'''	
predicting the results of the test set	
'''

y_pred = ann.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
