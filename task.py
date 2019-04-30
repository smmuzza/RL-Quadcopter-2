import numpy as np
from physics_sim import PhysicsSim

class TaskDefault():
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
        self.action_repeat = 3 # (SMM) default 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0 # (SMM), make this larger than 0 to enforce  all rotors on, default 0 
        self.action_high = 900
        self.action_size = 4 # (SMM) default 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""

        # original distance based reward 
        return 1. - .3 * (abs(self.sim.pose[:3] - self.target_pos)).sum()

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


class TaskFlyUp():
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
 
        upReward = 0    
        z = self.sim.pose[2]   
        vz = self.sim.v[2]
        zg = self.target_pos[2]

        if vz > 0:
            upReward += 0.1*(z + vz)
        else:
            upReward += 0.1*(z - vz) # vz is negative so this is a penality               
        
        reward = np.clip(reward, -1000, 1000)
        
        return upReward

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


class TaskFlyTowardsGoal():
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
        self.action_repeat = 4 # (SMM) default 3, dt of 20ms per action repeat

        self.state_size = self.action_repeat * 6
        
        # how much is needed to make the drove fly up?
        # fixing the RPMs of all rotors to 
        # 1000 RPM goes up very quickly, too quickly to be stable
        # 500 RPM goes up at a more moderate pace
        # 100 RPM makes the drone go down after 1.4 seconds
        # 200 RPM in 1.5 seconds
        # 300 RPM crashes in 2.0 seconds
        # 400 RPM goes down to a height of ~7m in 5 seconds
        self.action_low = -500 # (SMM), make this larger than 0 to enforce  all rotors on, default 0 
        self.action_high = 500 
        self.action_size = 4 # (SMM) default 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
 
        x0 = self.sim.init_pose[0]
        y0 = self.sim.init_pose[1]
        z0 = self.sim.init_pose[2]
        x = self.sim.pose[0]
        y = self.sim.pose[1]
        z = self.sim.pose[2]
        xr = self.sim.pose[3] # rotation angle in radians abount x axis
        yr = self.sim.pose[4] # rotation angle in radians abount x axis 
        zr = self.sim.pose[5]
        vx = self.sim.v[0]
        vy = self.sim.v[1]
        vz = self.sim.v[2]
        vxr = self.sim.angular_v[0]
        vyr = self.sim.angular_v[1]
        vzr = self.sim.angular_v[2]
        axr = self.sim.angular_accels[0]
        ayr = self.sim.angular_accels[1]
        azr = self.sim.angular_accels[2]
        xg = self.target_pos[0]
        yg = self.target_pos[1]
        zg = self.target_pos[2]
        dx = x - xg
        dy = y - yg
        dz = z - zg
        
        # compute current and inital distances to the goal
        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        r0 = np.sqrt(np.power(x0-xg,2.) + np.power(y0-yg, 2.) + np.power(z0-zg, 2.))
        
        reward = 1/self.action_repeat - (0.001 * r**2) # distance penalty     
        
        # z direction correct factor
        if z < zg and vz > 0:
            reward *= (1.0 + 0.01 * abs(dz) * abs(vz)) # right direction, increase reward by 75%
        elif z > zg and vz < 0:
            reward *= (1.0 + 0.01 * abs(dz) * abs(vz)) # right direction, increase reward by 75%
        elif reward > 0: # wrong direction, cut positive reward %
            reward *= (1.0 - 0.01 * abs(dz) * abs(vz))
        elif reward < 0: # wrong direction, increase negative reward %   
            reward *= (1.0 + 0.01 * abs(dz) * abs(vz))

        # x direction correct factor
        if x < xg and vx > 0:
            reward *= (1.0 + 0.01 * abs(dx) * abs(vx)) # right direction, increase reward by 75%
        elif x > xg and vx < 0:
            reward *= (1.0 + 0.01 * abs(dx) * abs(vx)) # right direction, increase reward by 75%
        elif reward > 0: # wrong direction, cut positive reward %
            reward *= (1.0 - 0.01 * abs(dx) * abs(vx))
        elif reward < 0: # wrong direction, increase negative reward %   
            reward *= (1.0 + 0.01 * abs(dx) * abs(vx))

        # y direction correct factor
        if y < yg and vy > 0:
            reward *= (1.0 + 0.02 * abs(dy) * abs(vy)) # right direction, increase reward by 75%
        elif y > yg and vy < 0:
            reward *= (1.0 + 0.02 * abs(dy) * abs(vy)) # right direction, increase reward by 75%
        elif reward > 0: # wrong direction, cut positive reward %
            reward *= (1.0 - 0.02 * abs(dy) * abs(vy))
        elif reward < 0: # wrong direction, increase negative reward %   
            reward *= (1.0 + 0.02 * abs(dy) * abs(vy))

        # small velocity factor, want 1 mps or less
        # if velocity is < 2mps, there is no penality
        reward *= 5. / np.clip(abs(vx)-5, 5, 10)        
        reward *= 5. / np.clip(abs(vy)-5, 5, 10) 
        reward *= 5. / np.clip(abs(vz)-5, 5, 10) 
           
        reward = np.clip(reward, -3/self.action_repeat, 3/self.action_repeat)
        
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
        
        # adjust the starting position randomly, random re-starts
        # in order to help the agent not get suck in a minimum
#        self.sim.pose[0] = self.sim.pose[0] + np.random.randn()
#        self.sim.pose[1] = self.sim.pose[1] + np.random.randn()
#        self.sim.pose[2] = self.sim.pose[2] + np.random.randn()
        
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state    