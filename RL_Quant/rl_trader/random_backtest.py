# random_backtest.py
import numpy as np
from envs.market_env import make_env
from configs.config import CONFIG

def main():
    window_size = CONFIG.get("window_size", 50)
    print(f"[INFO] Loading environment (window_size={window_size}) ...")
    env = make_env(window_size=window_size)

    obs, _ = env.reset()
    done = False
    rewards = []

    while not done:
        # 随机动作，而不是用 PPO
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(float(reward))

    rewards = np.asarray(rewards, dtype=float)
    print("steps:", len(rewards))
    print("min reward:", rewards.min(), "max reward:", rewards.max(), "mean:", rewards.mean())
    print("first 20 rewards:", rewards[:20])
    print("last info:", info)  # 看看 info 里有什么，比如 total_profit / total_reward 等

    env.close()

if __name__ == "__main__":
    main()
