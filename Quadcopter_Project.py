# Close other sessions
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

# Setup GPU TF stability
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(
  allow_soft_placement=True, log_device_placement=True))

import pandas as pd
import numpy as np
from agents.policy_search import PolicySearch_Agent
from agents.agent import DDPG
from task import TaskDefault, TaskFlyUp, TaskFlyTowardsGoal
from runSimulation import runSimulation

# init task (reward structure), and agent
# simulation time and number of episodes
init_pose = np.array([0., 0., 10.0, 0., 0., 0.])
target_pose = np.array([10., 10., 30.]) #SMM original [0., 0., 10.]
simTime = 5 # make the sim run longer so the agent has more chance to adapt
num_episodes = 2500
task = TaskFlyTowardsGoal(init_pose=init_pose, target_pos=target_pose, runtime=simTime)
useDefault = False
my_agent = DDPG(task, useDefault) 
print(my_agent)
print(task)
print("init_pose: ", init_pose)
print("target_pose: ", target_pose)

# Run the simulation and save the results.
showPlotEachEpisode = False
file_output = 'my_agent.txt' # save my results

runSimulation(init_pose, target_pose, simTime, num_episodes, task, my_agent,\
              showPlotEachEpisode, file_output)

