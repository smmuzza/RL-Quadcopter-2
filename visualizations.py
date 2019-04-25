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
episode_rewards = results.groupby(['episode'])[['reward']].sum()
episodes = results.groupby(['episode'])


#print(episode_rewards)
plt.figure(1)
plt.plot(episode_rewards, label='sum rewards')
plt.legend()
plt.show()  
len(episode_rewards)
plt.figure(2)
plt.scatter(results['time'][500:1000], results['x'][500:1000], label='x')
plt.scatter(results['time'][500:1000], results['y'][500:1000], label='y')
plt.scatter(results['time'][500:1000], results['z'][500:1000], label='z')
plt.legend()
plt.show()  
   
