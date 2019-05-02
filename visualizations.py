# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 07:21:45 2019

@author: shane
"""


## TODO: Plot the rewards.

import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

# Load simulation results from the .csv file
results = pd.read_csv('my_agent.txt')

# Total rewards for each episode
episode_rewards_mean = results.groupby(['episode'])[['reward']].mean()
episode_rewards_sum = results.groupby(['episode'])[['reward']].sum()

smoothed_mean = episode_rewards_mean.rolling(100).mean() 
smoothed_sum = episode_rewards_sum.rolling(100).mean() 

#print(episode_rewards)
plt.figure(1)
plt.plot(episode_rewards_mean, label='mean rewards')
plt.plot(smoothed_mean, label='running mean')
plt.legend()
axes = plt.gca()
axes.set_ylim([-1000,4000])
plt.show()  

# plot the 
plt.figure(2)
plt.plot(episode_rewards_sum, label='sum rewards')
plt.plot(smoothed_sum, label='running mean')
plt.legend()
axes = plt.gca()
axes.set_ylim([-1000,60000])
plt.show()  

# plot first 500 results
plt.figure(3)
plt.scatter(results['time'][0:500], results['x'][0:500], label='x')
plt.scatter(results['time'][0:500], results['y'][0:500], label='y')
plt.scatter(results['time'][0:500], results['z'][0:500], label='z')
plt.legend()
plt.show()  


# plot last 500 results
plt.figure(3)
plt.scatter(results['time'][1500:2000], results['x'][1500:2000], label='x')
plt.scatter(results['time'][1500:2000], results['y'][1500:2000], label='y')
plt.scatter(results['time'][1500:2000], results['z'][1500:2000], label='z')
plt.legend()
plt.show()  
   
