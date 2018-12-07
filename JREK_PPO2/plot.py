'''
Plot the results of a run
Takes a file rewards.csv
'''

import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('rewards_jerk_ppo2.csv')
    
    plt.figure()
    plt.title('Reward over time')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.axvline(color='k')
    plt.axhline(color='k')
    plt.plot(df['timestep'], df['reward'], color='r')
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    main()
