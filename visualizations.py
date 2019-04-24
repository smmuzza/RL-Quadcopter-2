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

# Plot the reward
#print(results)
plt.clf()
plt.figure(0)
plt.scatter(results.episode, results.reward)

# Total rewards for each episode
#episode_rewards = results.groupby(['episode'])[['reward']].sum() 
episode_rewards = results.groupby(['episode'])[['reward']].sum()
print(episode_rewards)
plt.figure(1)
plt.plot(episode_rewards)
