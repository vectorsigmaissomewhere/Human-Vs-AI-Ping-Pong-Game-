# train_agent.py
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from pong_env import PongEnv

def make_env():
    return PongEnv(max_steps=1000)

if __name__ == "__main__":
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    # vectorized environment
    env = DummyVecEnv([make_env])
    # optionally normalize observations/returns
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tb_logs/",
        policy_kwargs=dict(net_arch=[dict(pi=[64,64], vf=[64,64])]),
        learning_rate=3e-4,
        batch_size=64,
    )

    # callbacks: save checkpoints and evaluation
    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=save_dir, name_prefix="ppo_pong")
    # optional EvalCallback can be added if you create a separate eval env

    total_timesteps = 1_000_000  # change per your compute budget
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    model.save(os.path.join(save_dir, "ppo_pong_final"))
    env.save(os.path.join(save_dir, "vecnormalize.pkl"))  # save normalization stats
    print("Training finished and model saved.")
