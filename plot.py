'''
Plot the results of a run
Takes a file rewards.csv
'''

import pandas as pd
import matplotlib.pyplot as plt

def main():
    df_ppo2 = pd.read_csv('PPO2/rewards_ppo2.csv')
    df_jerk = pd.read_csv('jerk/rewards_jerk_original.csv')
    
    plt.figure()
    plt.title('Reward over time')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.axvline(color='k')
    plt.axhline(color='k')
    plt.plot(df_ppo2['timestep'], df_ppo2['reward'], color='r', label='PPO2')
    plt.plot(df_jerk['timestep'], df_jerk['reward'], color='b', label='JERK')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    main()
