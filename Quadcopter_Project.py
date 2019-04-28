## TODO: Train your agent here.

def runSimulation(init_pose, target_pose, simTime, num_episodes,\
                  task, my_agent, showPlotEachEpisode, file_output):

    import matplotlib.pyplot as plt
    import csv
   #%matplotlib inline
    labels = ['episode', 'time', 'reward', 'x', 'y', 'z', 'phi', 'theta', 'psi', 
              'x_velocity','y_velocity', 'z_velocity', 
              'phi_velocity', 'theta_velocity', 'psi_velocity', 
              'x_acceleration', 'y_acceleration', 'z_acceleration',
              'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4',
              'rotor_noise1', 'rotor_noise2', 'rotor_noise3', 'rotor_noise4']
   
    best_reward = -np.inf 
    best_episode = 0 
    
    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        
        for i_episode in range(1, num_episodes+1):
            state = my_agent.reset_episode() # start a new episode
            plotBestEpisode = False
    
            results = {x : [] for x in labels}
    
            while True:
                # run the 4 rotors at different RPMs
                action = my_agent.act(state) 
                next_state, reward, done = task.step(action)
                my_agent.step(action, reward, next_state, done)
                state = next_state
    
                # append to results
                to_write = [i_episode] + [task.sim.time] + [reward] + list(task.sim.pose)\
                           + list(task.sim.v) + list(task.sim.angular_v) + list(task.sim.linear_accel)\
                           + list(action) + list(my_agent.noise.state)
                for ii in range(len(labels)):
                    results[labels[ii]].append(to_write[ii])
                writer.writerow(to_write)
    
                if done:           
                    if my_agent.score > best_reward:
                        best_reward = my_agent.score
                        best_episode = i_episode  
                        plotBestEpisode = True
    
                    # count number of best episodes
                    # count number of times sim > 2 seconds 
    
                    print("\rEpi: {:4d}, score: {:7.3f} (best: {:7.3f}) in epi {}, cnt:{} \n".format(
                        i_episode, my_agent.score, best_reward, best_episode, my_agent.count), end="")  # [debug]
                    
                    if i_episode % 25 == 0 or showPlotEachEpisode or plotBestEpisode or i_episode > num_episodes - 3:
                        # plot linear info
                        plt.figure(i_episode)
                        plt.plot(results['time'], results['x'], label='x')
                        plt.plot(results['time'], results['y'], label='y')
                        plt.plot(results['time'], results['z'], label='z')
                        plt.plot(results['time'], results['x_velocity'], label='vx', linestyle='--')
                        plt.plot(results['time'], results['y_velocity'], label='vy', linestyle='--')
                        plt.plot(results['time'], results['z_velocity'], label='vz', linestyle='--')  
                        plt.plot(results['time'], results['x_acceleration'], label='ax', linestyle=':')
                        plt.plot(results['time'], results['y_acceleration'], label='ay', linestyle=':')
                        plt.plot(results['time'], results['z_acceleration'], label='az', linestyle=':')  
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
                        # plot actions and noise
                        plt.figure(i_episode + 3 * num_episodes + 1)
                        plt.plot(results['time'], results['rotor_speed1'], label='rotor1RPM', linestyle='-')
                        plt.plot(results['time'], results['rotor_speed2'], label='rotor1RPM', linestyle='-')
                        plt.plot(results['time'], results['rotor_speed3'], label='rotor1RPM', linestyle='-')
                        plt.plot(results['time'], results['rotor_speed4'], label='rotor1RPM', linestyle='-')
                        plt.plot(results['time'], results['rotor_noise1'], label='rotorNoise', linestyle='--')
                        plt.plot(results['time'], results['rotor_noise2'], label='rotorNoise', linestyle='--')
                        plt.plot(results['time'], results['rotor_noise3'], label='rotorNoise', linestyle='--')
                        plt.plot(results['time'], results['rotor_noise4'], label='rotorNoise', linestyle='--')
                        plt.legend()
                        plt.show()                      
                        # plot rewards
                        plt.figure(i_episode + 4 * num_episodes + 1)
                        plt.plot(results['time'], results['reward'], label='reward', linestyle='-')
                        plt.legend()
                        plt.show()  
    
                    break
            sys.stdout.flush()
    return print("completed simulation\n")

## Use this loop as a testing ground, focus on plotting each episode
## in this way we can see how the agent is doing each episode, instead
## of at the end of all episodes, therefore can stop early if no 
## good behavior seems to be happening

# Close other sessions
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

# Setup GPU TF stability
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(
  allow_soft_placement=True, log_device_placement=True))

import sys
import pandas as pd
import numpy as np
from agents.policy_search import PolicySearch_Agent
from agents.agent import DDPG
from task import TaskDefault, TaskFlyUp, TaskFlyTowardsGoal

# init task (reward structure), and agent
# simulation time and number of episodes
init_pose = np.array([0., 0., 10.0, 0., 0., 0.])
target_pose = np.array([0., 0., 10.]) #SMM original [0., 0., 10.]
simTime = 5 # make the sim run longer so the agent has more chance to adapt
num_episodes = 1000
task = TaskDefault(init_pose=init_pose, target_pos=target_pose, runtime=simTime)
useDefault = True
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

