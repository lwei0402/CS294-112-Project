import retro
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from retro_contest.local import make

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

# env = make(game='SonicTheHedgehog-Genesis', state='StarLightZone.Act3')
env = make(game='SonicTheHedgehog-Genesis', state = 'GreenHillZone.Act1')
env.seed(0)
env = DummyVecEnv([lambda: env])
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
obs = env.reset()
rewards_total = []

while iteration < 20:
    action, _info = model.predict(obs)    
    obs, rewards, dones, info = env.step(action)
    rewards_total.append(rewards)
    if dones:
        iteration+=1
        print(sum(rewards_total))
        rewards_total = []
    # env.render()