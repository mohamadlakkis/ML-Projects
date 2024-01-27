# lecture 208 to understand how the simulation works
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
'''
implementing UCB 
'''
N = 10000 # total number of user
d = 10 # number of ads
ads_selected = []  # will become then a list of 10000 ads of the 10000 users
numbers_of_selection = [0]*d # the numbers of the ad i was selcted
sums_of_rewards = [0]*d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if numbers_of_selection[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selection[i]
            delta_i = math.sqrt((3/2)*math.log(n + 1) / numbers_of_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selection[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward
'''
visualising the results
'''
plt.hist(ads_selected)
plt.title('histograme of ads selection')
plt.xlabel('ads')
plt.ylabel('Number of times each ads was selected')
plt.show()
#after that we need to see the minumin round of N that can work and comapre it to thonpson Sampling
