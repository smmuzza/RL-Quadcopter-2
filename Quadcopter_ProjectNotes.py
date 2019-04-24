# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 17:57:39 2019

@author: shane
"""

        # reward if velocity direction is towards the goal
        if 0:
        velDirectionRewardFactor = 0.01
        xVelDirection = self.target_pos[0] - self.sim.pose[0] # find if positive or negative direction
        if xVelDirection > 0 and self.sim.v[0] > 0:
            reward = reward + velDirectionRewardFactor * self.sim.v[0]
        elif xVelDirection < 0 and self.sim.v[0] < 0:     
            reward = reward + velDirectionRewardFactor * -self.sim.v[0]
        else:
            reward = reward - 1 * self.sim.v[0]

        yVelDirection = self.target_pos[1] - self.sim.pose[1]
        if yVelDirection > 0 and self.sim.v[1] > 0:
            reward = reward + velDirectionRewardFactor * self.sim.v[1]
        elif yVelDirection < 0 and self.sim.v[1] < 0:
            reward = reward + velDirectionRewardFactor * -self.sim.v[1]
        else:
            reward = reward - 1 * self.sim.v[1]
            
        zVelDirection = self.target_pos[2] - self.sim.pose[2]
        if zVelDirection > 0 and self.sim.v[2] > 0:
            reward = reward + velDirectionRewardFactor * self.sim.v[2]
        elif zVelDirection < 0 and self.sim.v[2] < 0:
            reward = reward + velDirectionRewardFactor * -self.sim.v[2]     
        else:
            reward = reward - 1 * self.sim.v[2]
