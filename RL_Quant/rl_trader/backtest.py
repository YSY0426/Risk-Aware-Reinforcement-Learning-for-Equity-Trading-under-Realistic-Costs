# backtest.py
"""
Backtest a trained PPO model on the custom trading environment (test split).
"""

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from gym_anytrading.datasets import STOCKS_GOOGL

from envs.custom_env import make_env
from configs.config import CONFIG
from utils.metrics import sharpe_ratio, max_drawdown


def main():
    window_size = CONFIG.get("window_size", 50)
    trading_cost = CONFIG.get("trading_cost", 0.001)
    risk_lambda = CONFIG.get("risk_lambda", 0.0)
    train_ratio = CONFIG.get("train_ratio", 0.7)
    model_path = CONFIG.get("model_path", "models/ppo_custom_env_googl.zip")

    # ===== 数据切分：后 30% 用于测试 =====
    df_full = STOCKS_GOOGL.copy().reset_index(drop=True)
    n = len(df_full)
    split_idx = int(n * train_ratio)
    df_test = df_full.iloc[split_idx:].reset_index(drop=True)

    print(f"[INFO] Full data length: {n}")
    print(f"[INFO] Test length: {len(df_test)} (ratio={1 - train_ratio:.2f})")

    env = make_env(
        df=df_test,
        window_size=window_size,
        trading_cost=trading_cost,
        risk_lambda=risk_lambda,
    )

    print(f"[INFO] Loading model from {model_path} ...")
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False

    rewards = []
    equity = [1.0]  # 初始资金

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rewards.append(float(reward))
        equity.append(float(info["equity"]))

    rewards = np.asarray(rewards, dtype=float)
    equity = np.asarray(equity, dtype=float)
    returns = equity[1:] / equity[:-1] - 1.0
    T = len(returns)

    print("---- DEBUG: rewards ----")
    print("first 10 rewards:", rewards[:10])
    print("min reward:", rewards.min(), "max reward:", rewards.max(), "mean:", rewards.mean())

    print("---- DEBUG: returns ----")
    print("first 10 returns:", returns[:10])
    print("min return:", returns.min(), "max return:", returns.max(), "mean:", returns.mean())

    sr = sharpe_ratio(returns)
    mdd = max_drawdown(equity)

    if T > 0:
        ann_ret = equity[-1] ** (252.0 / T) - 1.0
        ann_vol = returns.std(ddof=1) * np.sqrt(252.0)
    else:
        ann_ret = 0.0
        ann_vol = 0.0

    print(f"[RESULT] Steps (test period): {T}")
    print(f"[RESULT] Final equity: {equity[-1]:.4f}")
    print(f"[RESULT] Total net return (approx): {equity[-1] - 1.0:.6f}")
    print(f"[RESULT] Annualized return (approx): {ann_ret:.2%}")
    print(f"[RESULT] Annualized volatility (approx): {ann_vol:.2%}")
    print(f"[RESULT] Sharpe ratio (approx): {sr:.6f}")
    print(f"[RESULT] Max drawdown (approx): {mdd:.4%}")

    # 画测试区间的权益曲线
    plt.figure()
    plt.plot(equity)
    plt.title("Equity Curve (Test Period, Custom Env, Net of Cost)")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
