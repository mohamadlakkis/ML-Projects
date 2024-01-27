# people who bought also bought
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)#to tell 1 row is included
transcation = []
for i in range(0, 7501):
    transcation.append([str(dataset.values[i, j]) for j in range(0, 20)])
'''
training the eclat model on the whole dataset
'''
from apyori import apriori
rules = apriori(transactions=transcation, min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2,max_length=2) # min support = 3(min)*7(days)/7501(transaction)
'''
######
visualising the results
######
'''
'''
displaying the 1st result coming directly from the output of the apriori function
'''
results = list(rules)
print(results)
'''
putting the results well organised into panda DataFrame
'''
def inspect(results):
    lhs = [tuple(result[2][0][0])[0]for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1]for result in results]
    return list(zip(lhs,rhs,supports))
resultsinDataFrame = pd.DataFrame(inspect(results),columns=['product 1 ','product2','Support'])
'''
Displaying the results sorted by descending support
'''
print(resultsinDataFrame.nlargest(n = 10,columns='Support'))
