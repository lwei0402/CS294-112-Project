'''
Plot the results of a run
Takes a file rewards.csv
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('rewards_jerk.csv')
    print('mean', np.mean(df['reward']))
    print('std', np.std(df['reward']))
    plt.figure()
    plt.title('Reward over time')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.axvline(color='k')
    plt.axhline(color='k')
    plt.plot(df['timestep'], df['reward'], color='r', label='reward')
    
    plt.plot(df['timestep'], df['score'] * 200, color='g', label='score')
    plt.plot(df['timestep'], df['x'] * 1, color='b', label='x')
    plt.plot(df['timestep'], df['rings'] * 100, color='m', label='rings')
    plt.plot(df['timestep'], df['screen_x'] * 1, color='y', label='screen_x')
    plt.legend()
    plt.tight_layout()
    
    plt.figure()
    plt.title('Trajectory data')
    plt.xlabel('t')
    plt.grid(True)
    plt.plot(df['timestep'], df['x'], color='b', label='x')
    plt.plot(df['timestep'], df['screen_x'], color='c', label='screen_x')
    plt.plot(df['timestep'], df['y'], color='r', label='y')
    plt.plot(df['timestep'], df['screen_y'], color='m', label='screen_y')
    plt.legend()
    plt.tight_layout()
    
    plt.figure()
    plt.title('Trajectory')
    plt.xlabel('t')
    plt.grid(True)
    plt.plot(df['x'], df['y'][0] - df['y'], color='b')
    plt.legend()
    plt.tight_layout()
    
    plt.show()
    
if __name__ == '__main__':
    main()
