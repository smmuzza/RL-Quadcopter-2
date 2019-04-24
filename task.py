import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 

        #in order to make the problems approximately fully observable in the high dimensional 
        #environment we used action repeats. For each timestep of the agent, we step the simulation 
        #3 timesteps, repeating the agent’s action and rendering each time. Thus the observation 
        #reported to the agent contains9featuremaps(theRGBofeachofthe3renderings)
        #which allows the agent to infer velocities using the differences between frames
        self.action_repeat = 6 # (SMM) default 3

        self.state_size = self.action_repeat * 6
        self.action_low = 000 # (SMM), make this larger than 0 to enforce  all rotors on, default 0 
        self.action_high = 900
        self.action_size = 4 # (SMM) default 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
 
        # reset reward
        reward = 0
               
        # penalize large linear velocities
        #reward = reward - 0.001 * np.power(self.sim.v[0], 2.)
        #reward = reward - 0.001 * np.power(self.sim.v[1], 2.)
        #reward = reward - 0.001 * np.power(self.sim.v[2], 2.)

        # penalize large angles
        #reward = reward - 0.001 * np.power(self.sim.pose[3], 2.)
        #reward = reward - 0.001 * np.power(self.sim.pose[4], 2.)
        #reward = reward - 0.001 * np.power(self.sim.pose[5], 2.)

        # penalize large angular velocities (like from spinning)
        #reward = reward - 0.001 * np.power(self.sim.angular_v[0], 1.)
        #reward = reward - 0.001 * np.power(self.sim.angular_v[1], 1.)
        #reward = reward - 0.001 * np.power(self.sim.angular_v[2], 1.)

        # penalize large angular accelerations, reward smooth changes in behavior
        #reward = reward - 0.001 * np.power(self.sim.angular_accels[0], 1.)
        #reward = reward - 0.001 * np.power(self.sim.angular_accels[1], 1.)
        #reward = reward - 0.001 * np.power(self.sim.angular_accels[2], 1.)
               
        # reward if currently coming closer to goal positoon
        # in this case, the position delta should be prediced to decrease
        # if delta decreases give a positive reward, otherwise negative
        # this code seems to be the cause of getting stuck on the ground a lot
        #dt = 0.01 # predict for a small time
        #delta = abs(self.target_pos[0] - self.sim.pose[0])
        #deltapred = abs(self.target_pos[0] - (self.sim.pose[0] + self.sim.v[0] * dt))
        #reward = reward + (delta - deltapred) 
        #delta = abs(self.target_pos[1] - self.sim.pose[1])
        #deltapred = abs(self.target_pos[1] - (self.sim.pose[1] + self.sim.v[1] * dt))
        #reward = reward + (delta - deltapred)
        #delta = abs(self.target_pos[2] - self.sim.pose[2])
        #deltapred = abs(self.target_pos[2] - (self.sim.pose[2] + self.sim.v[2] * dt))
        #reward = reward + (delta - deltapred)
               
        # reward survival
        survivalReward = 0.001 * (self.sim.time)

        # reward going up if the agent is below Z of the goal
        upReward = 0.
        if self.sim.pose[2] <  self.target_pos[2] and self.sim.v[2] > 0:
            upReward = 0.01
        
        # use starting distance as the default reward
        distanceNow = np.linalg.norm(self.sim.pose[:3] - self.target_pos)   
        distanceStarting = np.linalg.norm(self.sim.init_pose[:3] - self.target_pos)
        distanceReward = 0.002 * (distanceStarting - distanceNow)
        
        # compute final reward
        reward = distanceReward + survivalReward + upReward
        
        # original distance based reward
        #reward = 1. - .3 * (abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state