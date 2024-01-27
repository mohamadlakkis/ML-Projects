import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
'''
implementing thompson Sampling
'''
from random import betavariate
N = 500
d = 10
ads_selected = []
numbers_of_reward_1 = [0] * 10
numbers_of_reward_0 = [0] * 10
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = betavariate(numbers_of_reward_1[i] + 1, numbers_of_reward_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_reward_1[ad] = numbers_of_reward_1[ad] + 1
    else:
        numbers_of_reward_0[ad] = numbers_of_reward_0[ad] + 1
    total_reward = total_reward + reward
'''
visualising the results - histogram
'''
plt.hist(ads_selected)
plt.title('histogram of ads selection')
plt.xlabel('ads')
plt.ylabel('Number of times each ads was selected')
plt.show()