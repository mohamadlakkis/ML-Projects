import pandas as pd
s = pd.Series(['jhon','mohamad','jad','elena']) # will give us a list with each index
print(s)
print(s[0])
print(s[1: 4])
print(s[[1,2,3]])
param = s != 'jhon'
print(s[param]) # to remove a specific item
d = {'New york': 1300,'chicago': 600,'San diego':950,'saida':1900}
cities = pd.Series(d) # in case of a dict it will give us the key not the index
print(cities)
print(cities[['chicago','saida']]) # to select a specific item
print(cities[cities < 1000])
cities[cities < 1000] = 80000 # to change all the cities below 1000 to 80000
print(cities)
'''
reading files
'''
titanic = pd.read_csv('Titanic.txt', sep='\t')
print(titanic.head())
print(titanic['age']) # will print the age collum
print('\n'*100)
print(titanic[['sex','age']])
print('\n'*100)
print(titanic[(titanic.age=='child')& (titanic.pclass=='2nd')]) # to add some conditions on what we want to print
print(titanic[320:329])
by_age = titanic.groupby('age') # to groupe by something specific in here we will see how many adults and children
print(by_age.size())