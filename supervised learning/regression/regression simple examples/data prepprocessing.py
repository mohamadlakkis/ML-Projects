#what you want to predict is the dependent variable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''
importing the Data set
'''
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values #to locate index : means arrange
y = dataset.iloc[:,-1].values
print(x)
print(y)
print(100*'\n')

'''
taking care of missing data
'''
## method 1 delete the entire row but if you dint have alot of data you may see errors
##method2 get the average of the row
from sklearn.impute import SimpleImputer #sklearn is a set of modules to machine lerning
imputer=SimpleImputer(missing_values=np.nan , strategy='mean')
imputer.fit(x[:, 1:3]) ## the fit method is to connect iputer to the table by choosing the row of the missing values
x[:, 1:3] = imputer.transform(x[:, 1:3]) ## the trnsform is to put the values in the table
print(x) #3 to check
'''
encoding categorial data # we need to replace all the letters wirh number
'''
# there is 3 != columms in country france ,geramany,spain so we need to do from this row 3 rows
#####encoding independant variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

##reminder(pssthrough) is too keep the coloums that we wont use them to missing data
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder='passthrough') # transformers we need to specify what kind of transfomation (endoding) and kind and its one hut coding and the index
### now we need to connect it to the table
x = np.array(ct.fit_transform(x)) ## we need to force this to give us an numpy array by using np.array
print(100*'\n')
print(x)
######encoding the dependent varibale
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() ##we dont need to specify anuthing in () beacuse y is only one colum and we dnt have to transforme colums
y = le.fit_transform(y) #we dont have to put as np eray beacuse the y dont need to be
print('up x down y')
print(y)
'''
splitting the dataset into Training set and Test set
'''
from sklearn.model_selection import train_test_split
x_train, x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=1)
print(100*'\n')
print(x_train)
print('up x train')
print(x_test)
print('up x test')
print(y_train)
print('y up train')
print(y_test)
print('y test')
'''
feature scalling
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
## we need to apply our squalling only for nume]) ##or tou can [:, 3:]  ## we are using standardisation view the  inf notebook
x_test[:, 3:5] = sc.transform(x_test[:,3:5])# values not for encoding
x_train[:,3:5] = sc.fit_transform(x_train[:,3:5]) ##we need to apply the same squale and thats why we didnt use fit
print(x_test)
print(x_train)
