# === NEW TEST_TRADING ===

import gymnasium as gym      # 用 gymnasium 替代旧的 gym
import gym_anytrading        # 导入后会自动注册 'stocks-v0' 等环境
from gym_anytrading.datasets import STOCKS_GOOGL
from stable_baselines3 import PPO


def main():
    print("Creating trading environment ...")

    # 1. 使用 gym-anytrading 自带的谷歌股票数据
    df = STOCKS_GOOGL

    window_size = 50
    frame_bound = (50, len(df))  # 从第 50 条数据开始，到最后一条

    # 2. 创建交易环境：stocks-v0
    env = gym.make(
        "stocks-v0",
        df=df,
        frame_bound=frame_bound,
        window_size=window_size,
    )

    print("Env created, start training PPO ...")

    # 3. 创建 PPO 模型并训练
    model = PPO("MlpPolicy", env, verbose=1)

    # 训练 10,000 步，时间大概几十秒
    model.learn(total_timesteps=10_000)

    env.close()
    print("Training finished successfully!")


if __name__ == "__main__":
    main()
