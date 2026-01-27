# train.py
"""
Train a PPO agent on the custom trading environment.
"""

import os

from stable_baselines3 import PPO

from envs.custom_env import make_env
from configs.config import CONFIG


def main():
    os.makedirs("models", exist_ok=True)

    window_size = CONFIG.get("window_size", 50)
    trading_cost = CONFIG.get("trading_cost", 0.001)
    total_timesteps = CONFIG.get("total_timesteps", 50_000)
    policy = CONFIG.get("policy", "MlpPolicy")
    model_path = CONFIG.get("model_path", "models/ppo_custom_env.zip")

    print(f"[INFO] Creating custom env: window_size={window_size}, trading_cost={trading_cost} ...")
    env = make_env(window_size=window_size, trading_cost=trading_cost)

    print("[INFO] Creating PPO model ...")
    model = PPO(policy, env, verbose=1)

    print(f"[INFO] Start training for {total_timesteps} timesteps ...")
    model.learn(total_timesteps=total_timesteps)

    print(f"[INFO] Saving model to {model_path} ...")
    model.save(model_path)

    env.close()
    print("[INFO] Training finished successfully!")


if __name__ == "__main__":
    main()
