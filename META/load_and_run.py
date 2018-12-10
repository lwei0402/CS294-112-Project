import tensorflow as tf
import retro
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from retro_contest.local import make
import ppo2ttifrutti
import ppo2ttifrutti_policies as policies
import ppo2ttifrutti_sonic_env as env
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
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
# def play(policy, env, update):

#     # Get state_space and action_space
#     ob_space = env.observation_space
#     ac_space = env.action_space

#     # Instantiate the model object (that creates step_model and train_model)
#     model = Model(policy=policy,
#                 ob_space=ob_space,
#                 action_space=ac_space,
#                 nenvs=1,
#                 nsteps=1,
#                 ent_coef=0,
#                 vf_coef=0,
#                 max_grad_norm=0)
    
#     # Load the model
#     load_path = "./models/"+ str(update) + "/model.ckpt"
#     print(load_path)

#     obs = env.reset()

#     # Play
#     score = 0
#     done = False

#     while done == False:
#         # Get the action
#         actions, values, _ = model.step(obs)
        
#         # Take actions in env and look the results
#         obs, rewards, done, info = env.step(actions)
        
#         score += rewards
    
#         env.render()
        
    # print("Score ", score)
    # env.close()
config = tf.ConfigProto()
    #os.environ["CUDA_VISIBLE_DEVICES"]="-1"
config.gpu_options.allow_growth = True # pylint: disable=E1101

# with tf.Session(config=config):
#     ppo2ttifrutti.learn(policy=,
#                         env=SubprocVecEnv([env.make_train_0]),
#                         nsteps=2048,
#                         nminibatches=16,
#                         lam=0.95,
#                         gamma=0.99,
#                         noptepochs=4,
#                         log_interval=1,
#                         ent_coef=0.01,
#                         lr=lambda _: 2e-4,
#                         cliprange=lambda _: 0.1,
#                         total_timesteps=int(1),
#                         save_interval=25)
env=SubprocVecEnv([env.make_val_1])
vf_coef=0.5
nsteps=2048
nminibatches=16
ent_coef=0.01
max_grad_norm=0.5
nenvs = env.num_envs
ob_space = env.observation_space
ac_space = env.action_space
nbatch = nenvs * nsteps
nbatch_train = nbatch // nminibatches
assert nbatch % nminibatches == 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

config.gpu_options.allow_growth = True
with tf.Session(config=config):
    make_model = lambda : ppo2ttifrutti.Model(policy=policies.CnnPolicy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)
    model = make_model()
    model.load('checkpoints/00475')
    iteration = 0
    obs = env.reset()
    rewards_total = []
    dones = 0
    states = None

    while iteration < 20:
        actions, values, states, neglogpacs = model.step(obs, states, dones)
        obs, rewards, dones, info = env.step(actions)
        rewards_total.append(rewards)
        if dones:
            iteration+=1
            print(sum(rewards_total)*100)
            rewards_total = []
        env.render()