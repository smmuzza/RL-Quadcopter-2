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
episodes = results.groupby(['time'])
episodes = results.groupby(['reward'])


#print(episode_rewards)
plt.figure(1)
plt.plot(episode_rewards_mean, label='mean rewards')
plt.legend()
axes = plt.gca()
axes.set_ylim([-50,50])
plt.show()  

# plot the 
plt.figure(2)
plt.plot(episode_rewards_sum, label='sum rewards')
plt.legend()
axes = plt.gca()
axes.set_ylim([-1000,1000])
plt.show()  

# plot last 500 results
plt.figure(3)
plt.scatter(results['time'][500:1000], results['x'][500:1000], label='x')
plt.scatter(results['time'][500:1000], results['y'][500:1000], label='y')
plt.scatter(results['time'][500:1000], results['z'][500:1000], label='z')
plt.legend()
plt.show()  
   
