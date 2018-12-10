from retro import make
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
import numpy as  np
import pandas as pd

# SpringYardZone.Act3
# SpringYardZone.Act2
# GreenHillZone.Act3
# GreenHillZone.Act1
# StarLightZone.Act2
# StarLightZone.Act1
# MarbleZone.Act2
# MarbleZone.Act1
# MarbleZone.Act3
# ScrapBrainZone.Act2
# LabyrinthZone.Act2
# LabyrinthZone.Act1
# LabyrinthZone.Act3

# SpringYardZone.Act1
# GreenHillZone.Act2
# StarLightZone.Act3
# ScrapBrainZone.Act1

TOTAL_TIMESTEPS = int(100000)
render = True

# save reward and timestep to csv    
def save(filename, time, reward):
    print('Saving to file:', filename)
    t = np.array(time)
    r = np.array(reward)
    recorded_data = np.stack((t, r), axis=1)
    df = pd.DataFrame(recorded_data)
    df.to_csv(filename, index=False, header=['timestep', 'reward'])

print('Loading environment')

# env = make(game='SonicTheHedgehog-Genesis', state='StarLightZone.Act3')
env = make(game='SonicTheHedgehog-Genesis', state = 'GreenHillZone.Act1')
env.seed(0)
env = DummyVecEnv([lambda: env])

print('Loading model')

model = PPO2(policy=CnnPolicy, 
			 env=env,
             n_steps=4096,
             nminibatches=8,
             lam=0.95,
             gamma=0.99,
             noptepochs=3,
             ent_coef=0.01,
             learning_rate=lambda _: 2e-4,
             cliprange=lambda _: 0.2, 
			 verbose=1)

model = PPO2.load("PPO2_v1_1mil_0.1clip", env=env)
iteration = 0
timestep = 0
obs = env.reset()
rewards_total = []
complete_time_history = []
complete_reward_history = []
print('Running model')

try:
	while iteration < 20 and timestep < TOTAL_TIMESTEPS:
		action, _info = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		timestep = timestep + 1
		rewards_total.append(rewards)
		complete_time_history.append(timestep)
		complete_reward_history.append(int(sum(rewards_total)))
		
		if dones:
			iteration+=1
			print(sum(rewards_total))
			rewards_total = []
			
		if render: env.render()
		
		if timestep % 1000 == 0:
			print(timestep, sum(rewards_total))
except KeyboardInterrupt:
	pass
	
if render: env.render(close=True)
save('rewards_ppo2.csv', complete_time_history, complete_reward_history)
