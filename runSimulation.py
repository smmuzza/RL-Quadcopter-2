## Use this loop as a testing ground, focus on plotting each episode
## in this way we can see how the agent is doing each episode, instead
## of at the end of all episodes, therefore can stop early if no 
## good behavior seems to be happening

import sys
import pandas as pd
import numpy as np

def runSimulation(init_pose, target_pose, simTime, num_episodes,\
                  task, my_agent, showPlotEachEpisode, file_output):

    import matplotlib.pyplot as plt
    import csv
   #%matplotlib inline
    labels = ['episode', 'time', 'reward', 'x', 'y', 'z', 'phi', 'theta', 'psi', 
              'x_velocity','y_velocity', 'z_velocity', 
              'phi_velocity', 'theta_velocity', 'psi_velocity', 
              'x_acceleration', 'y_acceleration', 'z_acceleration',
              'phi_acceleration', 'theta_acceleration', 'psi_acceleration',
              'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4',
              'rotor_noise1', 'rotor_noise2', 'rotor_noise3', 'rotor_noise4']
   
    best_score = -9999
    best_episode = 0 
    best_episode_count = 0
    
    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        
        meanEpisodeRewards = []
        goalReachedEpisodeCount = 0
        
        for i_episode in range(1, num_episodes+1):
            state = my_agent.reset_episode() # start a new episode
            plotBestEpisode = False
    
            results = {x : [] for x in labels}
    
            while task.goalReachedCount <= 50: # due to action repeat, make this higher than 10*actionrepeat
                # run the 4 rotors at different RPMs
                action = my_agent.act(state) 
                next_state, reward, done = task.step(action)               
                my_agent.step(action, reward, next_state, done)
                state = next_state
    
                # append to results
                to_write = [i_episode] + [task.sim.time] + [reward] + list(task.sim.pose)\
                           + list(task.sim.v) + list(task.sim.angular_v)\
                           + list(task.sim.linear_accel) + list(task.sim.angular_accels)\
                           + list(action) + list(my_agent.noise.state)
                for ii in range(len(labels)):
                    results[labels[ii]].append(to_write[ii])
                writer.writerow(to_write)                
    
                if done:           
                    
                    if my_agent.best_score > best_score: 
                        best_score = my_agent.best_score
                        best_episode = i_episode
                        best_episode_count += 1
                        plotBestEpisode = True
                        
                    if goalReachedEpisodeCount < task.goalReachedCount:
                        goalReachedEpisodeCount = task.goalReachedCount    
                        plotBestEpisode = True                       
                    
                    meanEpisodeRewards.append(np.mean(results['reward']))
                    
                    print("\rEpi: {:4d}, score: {:7.5f} (best: {:7.5f}) in epi {}, BestEpiCnt: {}, goalCnt: {}\n".format(
                        i_episode, reward, best_score, best_episode, best_episode_count, task.goalReachedCount), end="")  # [debug]
                    
                    if i_episode % 50 == 0 or showPlotEachEpisode or plotBestEpisode or i_episode > num_episodes - 3:
                        # plot linear info
                        plt.figure(i_episode)
                        plt.plot(results['time'], results['x'], label='x')
                        plt.plot(results['time'], results['y'], label='y')
                        plt.plot(results['time'], results['z'], label='z')
                        plt.plot(results['time'], results['x_velocity'], label='vx', linestyle='-.')
                        plt.plot(results['time'], results['y_velocity'], label='vy', linestyle='-.')
                        plt.plot(results['time'], results['z_velocity'], label='vz', linestyle='-.')  
#                        plt.plot(results['time'], results['x_acceleration'], label='ax', linestyle=':')
#                        plt.plot(results['time'], results['y_acceleration'], label='ay', linestyle=':')
#                        plt.plot(results['time'], results['z_acceleration'], label='az', linestyle=':')  
                        plt.legend()
                        plt.show()  
                        
                        # plot angluar info
                        plt.figure(i_episode + 1 * num_episodes + 1)
                        plt.plot(results['time'], results['phi'], label='phi', linestyle='-')
                        plt.plot(results['time'], results['theta'], label='theta', linestyle='-')
                        plt.plot(results['time'], results['psi'], label='psi', linestyle='-')
                        plt.plot(results['time'], results['phi_velocity'], label='phi_v', linestyle='-.')
                        plt.plot(results['time'], results['theta_velocity'], label='theta_v', linestyle='-.')
                        plt.plot(results['time'], results['psi_velocity'], label='psi_v', linestyle='-.')
#                        plt.plot(results['time'], results['phi_acceleration'], label='phi_', linestyle=':')
#                        plt.plot(results['time'], results['theta_acceleration'], label='theta_a', linestyle=':')
#                        plt.plot(results['time'], results['psi_acceleration'], label='psi_a', linestyle=':')  
                        plt.legend()
                        plt.show() 
                        
                        # plot actions and noise
                        plt.figure(i_episode + 3 * num_episodes + 1)
                        plt.plot(results['time'], results['rotor_speed1'], label='rotor1RPM', linestyle='-')
                        plt.plot(results['time'], results['rotor_speed2'], label='rotor2RPM', linestyle='-')
                        plt.plot(results['time'], results['rotor_speed3'], label='rotor3RPM', linestyle='-')
                        plt.plot(results['time'], results['rotor_speed4'], label='rotor4RPM', linestyle='-')
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
    
                    # plot the mean rewards to monitor progress
                    if i_episode % 100 == 0 or plotBestEpisode:                       
                        # Total rewards for each episode
                        smoothed_mean = pd.DataFrame(meanEpisodeRewards).rolling(100).mean() 
                        plt.figure(i_episode + 4 * num_episodes + 1)
                        plt.plot(meanEpisodeRewards, label='mean rewards')
                        plt.plot(smoothed_mean, label='running mean')
                        plt.legend()
#                        axes = plt.gca()
#                        axes.set_ylim([-100,400])
                        plt.show()  

                    break
                
            sys.stdout.flush()
    return print("completed simulation\n")