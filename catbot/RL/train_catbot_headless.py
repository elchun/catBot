import os
import sys
import datetime
import gym
import numpy as np
import torch
from pydrake.all import StartMeshcat

from catbot.RL.catbot_rl_env import CatBotEnv

from psutil import cpu_count

num_cpu = int(cpu_count() / 2)

# Optional imports (these are heavy dependencies for just this one notebook)
sb3_available = False
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    sb3_available = True
except ImportError:
    print("stable_baselines3 not found")
    print("Consider 'pip3 install stable_baselines3'.")

def main():
    # -- ENV SETUP -- #
    gym.envs.register(
        id="CatBot-v0", entry_point="catbot.RL.catbot_rl_env:CatBotEnv"
    )

    # -- TRAINING -- #
    observations = "state"
    time_limit = 8
    checkpoint_timesteps = 800000
    num_checkpoints = 4
    # total time_steps is checkpoint_timesteps * num_checkpoints

    env = make_vec_env(
        CatBotEnv,
        n_envs=8,  # Was num_cpu
        seed=0,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "observations": observations,
            "time_limit": time_limit,
        },
    )

    # use_pretrained_model = False
    use_pretrained_model = True
    # model_zip_fname = "./models/230510_121121_PPO_C10_3.zip"
    model_zip_fname = "./checkpoints/230510_160600_PPO_P0_C13_sde_800000_steps.zip"
    print('starting training')
    tensorboard_log = "./ppo_cat_bot_logs"
    # tensorboard_log = None
    if use_pretrained_model:
        model = PPO.load(model_zip_fname, env)
    else:
        model = PPO(
            "MlpPolicy",
            # wrapped_env,
            env,
            verbose=0,
            n_steps=12,
            n_epochs=2,
            batch_size=32,
            learning_rate=1e-4,
            ent_coef=0.01,
            # use_sde=True,
            tensorboard_log=tensorboard_log)


    desc_str = "C13_sde"
    datetime_val = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    model_prefix = f'{datetime_val}_PPO_P{int(use_pretrained_model)}_{desc_str}'
    save_fname = f'./models/{model_prefix}_final.zip'

    checkpoint_callback = CheckpointCallback(
        save_freq=int(10000 / 8),
        save_path='./checkpoints/',
        name_prefix=model_prefix,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # -- TRAIN -- #
    model.learn(total_timesteps=checkpoint_timesteps, progress_bar=True, callback=checkpoint_callback)
    model.save(save_fname)
    print(f'Saved model to: {save_fname}')


if __name__ == "__main__":
    sys.exit(main())