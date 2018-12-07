import retro
import retro_contest
import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from retro_contest.local import make
from stable_baselines.common.vec_env import SubprocVecEnv


# def make1(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', discrete_actions=False, bk2dir=None):
#     use_restricted_actions = retro.Actions.FILTERED
#     if discrete_actions:
#         use_restricted_actions = retro.Actions.DISCRETE
#     try:
#         env = retro.make(game, state, scenario='contest', use_restricted_actions=use_restricted_actions)
#     except Exception:
#         env = retro.make(game, state, use_restricted_actions=use_restricted_actions)
#     if bk2dir:
#         env.auto_record(bk2dir)
#     env = retro_contest.StochasticFrameSkip(env, n=4, stickprob=0.25)
#     env = gym.wrappers.TimeLimit(env, max_episode_steps=4500)
#     return env

# def make2(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act2', discrete_actions=False, bk2dir=None):
#     use_restricted_actions = retro.Actions.FILTERED
#     if discrete_actions:
#         use_restricted_actions = retro.Actions.DISCRETE
#     try:
#         env = retro.make(game, state, scenario='contest', use_restricted_actions=use_restricted_actions)
#     except Exception:
#         env = retro.make(game, state, use_restricted_actions=use_restricted_actions)
#     if bk2dir:
#         env.auto_record(bk2dir)
#     env = retro_contest.StochasticFrameSkip(env, n=4, stickprob=0.25)
#     env = gym.wrappers.TimeLimit(env, max_episode_steps=4500)
#     return env

env = make(game='SonicTheHedgehog-Genesis')
# env = make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act3')
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
             cliprange=lambda _: 0.1, 
			 verbose=1)
model.learn(total_timesteps=1000000)

model.save("PPO2_v1_1mil_0.1clip")

iteration = 0
obs = env.reset()
rewards_total = []
while iteration<20:
    action, _info = model.predict(obs)    
    obs, rewards, dones, info = env.step(action)
    rewards_total.append(rewards)
    if dones:
        iteration += 1
        print(sum(rewards_total))
        rewards_total = []
    # env.render()