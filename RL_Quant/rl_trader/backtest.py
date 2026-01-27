# backtest.py
"""
Backtest a trained PPO model on the custom trading environment.
"""

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from envs.custom_env import make_env
from configs.config import CONFIG
from utils.metrics import sharpe_ratio, max_drawdown


def main():
    window_size = CONFIG.get("window_size", 50)
    trading_cost = CONFIG.get("trading_cost", 0.001)
    model_path = CONFIG.get("model_path", "models/ppo_custom_env.zip")

    print(f"[INFO] Loading custom env (window_size={window_size}, trading_cost={trading_cost}) ...")
    env = make_env(window_size=window_size, trading_cost=trading_cost)

    print(f"[INFO] Loading model from {model_path} ...")
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False

    rewards = []
    equity = [1.0]

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rewards.append(float(reward))
        equity.append(float(info["equity"]))  # 直接用 env 里算好的权益

    rewards = np.asarray(rewards, dtype=float)
    equity = np.asarray(equity, dtype=float)

    # 每步净收益率（已经包含成本）
    returns = equity[1:] / equity[:-1] - 1.0

    print("---- DEBUG: rewards ----")
    print("first 10 rewards:", rewards[:10])
    print("min reward:", rewards.min(), "max reward:", rewards.max(), "mean:", rewards.mean())

    print("---- DEBUG: returns ----")
    print("first 10 returns:", returns[:10])
    print("min return:", returns.min(), "max return:", returns.max(), "mean:", returns.mean())

    sr = sharpe_ratio(returns)
    mdd = max_drawdown(equity)

    print(f"[RESULT] Steps: {len(rewards)}")
    print(f"[RESULT] Total net return (approx): {equity[-1] - 1.0:.6f}")
    print(f"[RESULT] Sharpe ratio (approx): {sr:.6f}")
    print(f"[RESULT] Max drawdown (approx): {mdd:.4%}")

    # 画权益曲线
    plt.figure()
    plt.plot(equity)
    plt.title("Equity Curve (Custom Env, Net of Cost)")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
