# envs/custom_env.py
"""
A simple custom trading environment for one stock.

- Data: uses STOCKS_GOOGL from gym_anytrading as price series.
- Action: continuous target position in [-1, 1] (short to long).
- State: last `window_size` daily returns + current position.
- Reward: portfolio_return - trading_cost * turnover.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gym_anytrading.datasets import STOCKS_GOOGL


class CustomTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df=None, window_size: int = 50, trading_cost: float = 0.001, render_mode=None):
        super().__init__()

        # 使用示例数据，如果用户没有传入 df
        if df is None:
            df = STOCKS_GOOGL.copy()

        df = df.reset_index(drop=True)
        prices = df["Close"].astype(np.float32).values
        # 日收益率 r_t = (P_t / P_{t-1}) - 1
        self.returns = np.diff(prices) / prices[:-1]  # 长度: len(prices) - 1

        assert window_size < len(self.returns), "window_size 太大了，超过数据长度"

        self.window_size = window_size
        self.trading_cost = float(trading_cost)

        # 动作空间: 目标仓位 ∈ [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 状态: 最近 window_size 个收益率 + 当前仓位 (共 window_size + 1 维)
        obs_low = np.full((window_size + 1,), -np.inf, dtype=np.float32)
        obs_high = np.full((window_size + 1,), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # 内部状态
        self.current_step = None
        self.position = None
        self.prev_position = None
        self.equity = None

    # 生成观测
    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        window_returns = self.returns[start:end]  # 长度 = window_size
        obs = np.concatenate([window_returns, np.array([self.position], dtype=np.float32)], axis=0)
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.window_size  # 从有足够窗口的位置开始
        self.position = 0.0
        self.prev_position = 0.0
        self.equity = 1.0

        obs = self._get_observation()
        info = {"equity": self.equity, "position": self.position}
        return obs, info

    def step(self, action):
        # Gymnasium: action 可能是数组，取出单个数值
        if isinstance(action, (list, tuple, np.ndarray)):
            target_pos = float(action[0])
        else:
            target_pos = float(action)

        # 限制在 [-1, 1]
        target_pos = float(np.clip(target_pos, -1.0, 1.0))

        self.prev_position = self.position
        self.position = target_pos

        # 当前步使用 self.returns[self.current_step]
        idx = self.current_step
        assert idx < len(self.returns), "current_step 越界了"

        raw_ret = float(self.returns[idx])  # 基础资产日收益率
        # 组合收益（不含成本）：上一时刻持仓 * 当日收益率
        portfolio_return = self.prev_position * raw_ret

        # 换手率 = 仓位变化的绝对值
        turnover = abs(self.position - self.prev_position)
        cost = self.trading_cost * turnover

        # 奖励 = 净收益 = 组合收益 - 成本
        reward = portfolio_return - cost

        # 更新权益（用净收益近似复利）
        self.equity *= (1.0 + reward)

        # 前进一步
        self.current_step += 1
        terminated = self.current_step >= len(self.returns)  # 到头了就结束
        truncated = False

        obs = self._get_observation()
        info = {
            "raw_return": raw_ret,
            "portfolio_return": portfolio_return,
            "cost": cost,
            "equity": self.equity,
            "position": self.position,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        print(
            f"step={self.current_step}, equity={self.equity:.4f}, "
            f"position={self.position:.3f}"
        )


def make_env(window_size: int = 50, trading_cost: float = 0.001):
    """
    工厂函数：创建一个自定义交易环境实例。
    """
    return CustomTradingEnv(window_size=window_size, trading_cost=trading_cost)
