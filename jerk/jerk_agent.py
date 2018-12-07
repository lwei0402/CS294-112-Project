#!/usr/bin/env python

"""
A scripted agent called "Just Enough Retained Knowledge".
"""

import random

import gym
import numpy as np
import pandas as pd
from retro import make

EXPLOIT_BIAS = 0.25
TOTAL_TIMESTEPS = int(100000)
render = True

def main():
    """Run JERK on the attached environment."""
    env = make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', scenario='scenario.json')
    env = TrackedEnv(env)
    new_ep = True
    solutions = []
    env.reset()
    
    try:
        if render: env.render()
        print('Running agent for {} timesteps'.format(TOTAL_TIMESTEPS))
        
        while True:
            if env.total_steps_ever >= TOTAL_TIMESTEPS:
                break
                
            if new_ep:
                if (solutions and
                        random.random() < EXPLOIT_BIAS + env.total_steps_ever / TOTAL_TIMESTEPS):
                    print('Exploiting a previous solution')
                    solutions = sorted(solutions, key=lambda x: np.mean(x[0]))
                    best_pair = solutions[-1]
                    new_rew = exploit(env, best_pair[1])
                    best_pair[0].append(new_rew)
                    print('replayed best with reward %f' % new_rew)
                    continue
                else:
                    env.reset()
                    new_ep = False
            rew, new_ep = move(env, 100)
            if not new_ep and rew <= 0:
                print('backtracking due to negative reward: %f' % rew)
                _, new_ep = move(env, 50, left=True)
            if new_ep:
                print('Adding to solutions list')
                solutions.append(([max(env.reward_history)], env.best_sequence()))
    except KeyboardInterrupt:
        pass
    
    print('Ending agent')
    if render: env.render(close=True) # Needed to close render window without error
    env.save('rewards_jerk.csv')
    env.close()
    
def move(env, num_steps, left=False, jump_prob=1.0 / 10.0, jump_repeat=4):
    """
    Move right or left for a certain number of steps,
    jumping periodically.
    """
    total_rew = 0.0
    done = False
    steps_taken = 0
    jumping_steps_left = 0
    while not done and steps_taken < num_steps and env.total_steps_ever < TOTAL_TIMESTEPS:
        action = np.zeros((12,), dtype=np.bool)
        action[6] = left
        action[7] = not left
        if jumping_steps_left > 0:
            action[0] = True
            jumping_steps_left -= 1
        else:
            if random.random() < jump_prob:
                jumping_steps_left = jump_repeat - 1
                action[0] = True
        _, rew, done, _ = env.step(action)
        total_rew += rew
        steps_taken += 1
        if done:
            break
    return total_rew, done

def exploit(env, sequence):
    """
    Replay an action sequence; pad with NOPs if needed.

    Returns the final cumulative reward.
    """
    env.reset()
    done = False
    idx = 0
    while not done and env.total_steps_ever < TOTAL_TIMESTEPS:
        if idx >= len(sequence):
            _, _, done, _ = env.step(np.zeros((12,), dtype='bool'))
        else:
            _, _, done, _ = env.step(sequence[idx])
        idx += 1
    return env.total_reward

class TrackedEnv(gym.Wrapper):
    """
    An environment that tracks the current trajectory and
    the total number of timesteps ever taken.
    """
    def __init__(self, env):
        super(TrackedEnv, self).__init__(env)
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        self.total_steps_ever = 0
        self.complete_time_history = []
        self.complete_reward_history = []

    def best_sequence(self):
        """
        Get the prefix of the trajectory with the best
        cumulative reward.
        """
        max_cumulative = max(self.reward_history)
        for i, rew in enumerate(self.reward_history):
            if rew == max_cumulative:
                return self.action_history[:i+1]
        raise RuntimeError('unreachable')

    # pylint: disable=E0202
    def reset(self, **kwargs):
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.total_steps_ever += 1
        self.action_history.append(action.copy())
        obs, rew, done, info = self.env.step(action)
        if render: self.env.render()
        self.total_reward += rew
        self.reward_history.append(self.total_reward)
        self.complete_time_history.append(self.total_steps_ever)
        self.complete_reward_history.append(self.total_reward)
        #self.complete_reward_history.append(rew)
        
        if self.total_steps_ever % 1000 == 0:
            print('timestep {}: reward = {}'.format(self.total_steps_ever, self.total_reward))
            #print('timestep {}: reward = {}'.format(self.total_steps_ever, rew))
            
        return obs, rew, done, info
    
    # save reward and timestep to csv    
    def save(self, filename):
        print('Saving to file:', filename)
        t = np.array(self.complete_time_history)
        r = np.array(self.complete_reward_history)
        recorded_data = np.stack((t, r), axis=1)
        df = pd.DataFrame(recorded_data)
        df.to_csv(filename, index=False, header=['timestep', 'reward'])
        
if __name__ == '__main__':
    main()
