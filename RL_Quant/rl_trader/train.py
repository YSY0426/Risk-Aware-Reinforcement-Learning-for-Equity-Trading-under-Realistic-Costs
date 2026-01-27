# train.py
"""
Train a PPO agent on the custom trading environment (train split only).
"""

import os

from stable_baselines3 import PPO
from gym_anytrading.datasets import STOCKS_GOOGL

from envs.custom_env import make_env
from configs.config import CONFIG


def main():
    os.makedirs("models", exist_ok=True)

    window_size = CONFIG.get("window_size", 50)
    trading_cost = CONFIG.get("trading_cost", 0.001)
    risk_lambda = CONFIG.get("risk_lambda", 0.0)
    total_timesteps = CONFIG.get("total_timesteps", 50_000)
    policy = CONFIG.get("policy", "MlpPolicy")
    model_path = CONFIG.get("model_path", "models/ppo_custom_env_googl.zip")
    train_ratio = CONFIG.get("train_ratio", 0.7)
    seed = CONFIG.get("random_seed", 42)

    # ===== 数据切分：前 train_ratio 用于训练 =====
    df_full = STOCKS_GOOGL.copy().reset_index(drop=True)
    n = len(df_full)
    split_idx = int(n * train_ratio)
    df_train = df_full.iloc[:split_idx].reset_index(drop=True)

    print(f"[INFO] Full data length: {n}")
    print(f"[INFO] Train length: {len(df_train)} (ratio={train_ratio:.2f})")

    env = make_env(
        df=df_train,
        window_size=window_size,
        trading_cost=trading_cost,
        risk_lambda=risk_lambda,
    )

    print("[INFO] Creating PPO model ...")
    model = PPO(policy, env, verbose=1, seed=seed)

    print(f"[INFO] Start training for {total_timesteps} timesteps ...")
    model.learn(total_timesteps=total_timesteps)

    print(f"[INFO] Saving model to {model_path} ...")
    model.save(model_path)
    env.close()
    print("[INFO] Training finished successfully!")


if __name__ == "__main__":
    main()
