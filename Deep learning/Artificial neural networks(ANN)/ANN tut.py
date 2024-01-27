'''
####
data preprocessing
####
'''


import numpy as np
import pandas as pd
import tensorflow as tf
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

'''
encoding categorical data
'''

# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

# one hot encoding the geography column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# splitting the dataset into training set ad test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

'''#
feature scaling
'''

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

'''
####
building the ANN
####
'''

'''
initializing the ANN
'''

ann = tf.keras.models.Sequential()

'''
adding the input layer and the first hidden layer
'''

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))# units is nb of hidden layer#activation is what ac function we want to use

'''
adding the second hidden layer
'''

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

'''
adding the output layer
'''

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # we only need in this ex 1 neuron 1 or 2//sigmoid act function

'''
####
Training the output layer
####
'''

'''
compiling the ANN
'''

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

'''
training the ANN on the training set
'''

ann.fit(x_train, y_train, batch_size=32, epochs=100)

'''
####
making prediction and evaluating the model
####
'''

'''
predicting the result of a single observation
'''

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)# the out will be prob so if prob>0.5 will leave else stay

'''
predicting the test set result
'''

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

'''
making the confusion matrix
'''

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
