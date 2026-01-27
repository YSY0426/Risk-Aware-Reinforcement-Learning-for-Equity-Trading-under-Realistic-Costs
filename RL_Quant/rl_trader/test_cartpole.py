# test_cartpole.py
import gymnasium as gym
from stable_baselines3 import PPO

def main():
    # 创建 CartPole 环境
    env = gym.make("CartPole-v1")

    # 创建 PPO 模型
    model = PPO("MlpPolicy", env, verbose=1)

    # 训练 10,000 步（时间不长）
    model.learn(total_timesteps=10_000)

    # 训练完关闭环境
    env.close()

if __name__ == "__main__":
    main()
