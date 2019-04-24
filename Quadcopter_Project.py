## TODO: Train your agent here.

if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

import tensorflow as tf
#config = tf.ConfigProto()
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(
  allow_soft_placement=True, log_device_placement=True))

## Use this loop as a testing ground, focus on plotting each episode
## in this way we can see how the agent is doing each episode, instead
## of at the end of all episodes, therefore can stop early if no 
## good behavior seems to be happening

import sys
import pandas as pd
import numpy as np
from agents.agent import DDPG
from task import Task

import csv

# Init simulation
# scenario 1, try to go straight up
init_pose = np.array([0., 0., 0.0, 0., 0., 0.])
target_pose = np.array([0., 0., 10.]) #SMM original [0., 0., 10.]
simTime = 5 # make the sim run longer so the agent has more chance to adapt

# init task (reward structure), and agent
task = Task(init_pose=init_pose, target_pos=target_pose, runtime=simTime)
my_agent = DDPG(task) 
print(my_agent)

import matplotlib.pyplot as plt
#%matplotlib inline
labels = ['episode', 'time', 'reward', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
      'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
      'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']


# Run the simulation, and save the results.
num_episodes = 1500
best_reward = -np.inf 
best_episode = 0 
showPlotEachEpisode = False
file_output = 'my_agent.txt' # save my results
with open(file_output, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(labels)
    
    for i_episode in range(1, num_episodes+1):
        state = my_agent.reset_episode() # start a new episode
        plotBestEpisode = False

        # gather info for each episode to plot it
        xPos = []
        yPos = []
        zPos = []
        vx = []
        vy = []
        vz = []
        time = []

        results = {x : [] for x in labels}

        while True:
            # run the 4 rotors at different RPMs
            action = my_agent.act(state) 
            next_state, reward, done = task.step(action)
            my_agent.step(action, reward, next_state, done)
            state = next_state

            # append to indivdual variables
            xPos.append(task.sim.pose[0])
            yPos.append(task.sim.pose[1])
            zPos.append(task.sim.pose[2])
            vx.append(task.sim.v[0])
            vy.append(task.sim.v[1])
            vz.append(task.sim.v[2])
            time.append(task.sim.time)

            # append to results
            to_write = [i_episode] + [task.sim.time] + [reward] + list(task.sim.pose)\
                       + list(task.sim.v) + list(task.sim.angular_v) + list(action)
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            writer.writerow(to_write)

            if done:           
                if my_agent.score > best_reward:
                    best_reward = my_agent.score
                    best_episode = i_episode  
                    plotBestEpisode = True

                print("\rEpi = {:4d}, score = {:7.3f} (best = {:7.3f}) in epi {}".format(
                    i_episode, my_agent.score, my_agent.best_score, best_episode), end="")  # [debug]
                
                if showPlotEachEpisode or plotBestEpisode:
                    # plot linear info
                    plt.figure(i_episode)
                    plt.plot(time, xPos, label='x')
                    plt.plot(time, yPos, label='y')
                    plt.plot(time, zPos, label='z')
                    plt.plot(time, vx, label='vx', linestyle='--')
                    plt.plot(time, vy, label='vy', linestyle='--')
                    plt.plot(time, vz, label='vz', linestyle='--')  
                    plt.legend()
                    plt.show()  
                    # plot angluar info
                    plt.figure(i_episode + 1 * num_episodes + 1)
                    plt.plot(results['time'], results['phi'], label='phi', linestyle='-.')
                    plt.plot(results['time'], results['theta'], label='theta', linestyle='-.')
                    plt.plot(results['time'], results['psi'], label='psi', linestyle='-.')
                    plt.plot(results['time'], results['phi_velocity'], label='phi_velocity', linestyle=':')
                    plt.plot(results['time'], results['theta_velocity'], label='theta_velocity', linestyle=':')
                    plt.plot(results['time'], results['psi_velocity'], label='psi_velocity', linestyle=':')
                    plt.legend()
                    plt.show() 
                    # plot rewards
                    plt.figure(i_episode + 2 * num_episodes + 1)
                    plt.plot(results['time'], results['reward'], label='reward', linestyle='-')
                    plt.legend()
                    plt.show()  

                break
        sys.stdout.flush()