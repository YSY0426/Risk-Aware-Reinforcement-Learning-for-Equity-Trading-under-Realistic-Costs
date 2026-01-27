# envs/custom_env.py
"""
A simple custom trading environment for one stock.

- Data: by default uses STOCKS_GOOGL from gym_anytrading, but can accept a custom df.
- Action: continuous target position in [-1, 1] (short to long).
- State: last `window_size` daily returns + current position.
- Reward: portfolio_return - trading_cost * turnover - risk_lambda * |position|.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gym_anytrading.datasets import STOCKS_GOOGL


class CustomTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df=None,
        window_size: int = 50,
        trading_cost: float = 0.001,
        risk_lambda: float = 0.0,
        render_mode=None,
    ):
        super().__init__()

        # ===== 数据准备 =====
        if df is None:
            df = STOCKS_GOOGL.copy()

        df = df.reset_index(drop=True)
        prices = df["Close"].astype(np.float32).values

        # 日收益率 r_t = (P_t / P_{t-1}) - 1
        self.returns = np.diff(prices) / prices[:-1]  # len = len(prices) - 1
        self.returns = self.returns.astype(np.float32)

        assert window_size < len(self.returns), "window_size 太大，超过数据长度了"

        self.window_size = int(window_size)
        self.trading_cost = float(trading_cost)
        self.risk_lambda = float(risk_lambda)

        # ===== 动作空间: 目标仓位 [-1, 1] =====
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # ===== 状态: window_size 个收益 + 当前仓位 =====
        obs_dim = self.window_size + 1  # +1 for position
        obs_low = np.full((obs_dim,), -np.inf, dtype=np.float32)
        obs_high = np.full((obs_dim,), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # ===== 内部状态 =====
        self.current_step: int | None = None
        self.position: float | None = None
        self.prev_position: float | None = None
        self.equity: float | None = None
        self.max_equity: float | None = None

    # 生成观测
    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        window_returns = self.returns[start:end]  # 长度 = window_size
        obs = np.concatenate(
            [window_returns, np.array([self.position], dtype=np.float32)], axis=0
        )
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 从有足够窗口的位置开始
        self.current_step = self.window_size
        self.position = 0.0
        self.prev_position = 0.0
        self.equity = 1.0
        self.max_equity = 1.0

        obs = self._get_observation()
        info = {
            "equity": self.equity,
            "position": self.position,
        }
        return obs, info

    def step(self, action):
        # Gymnasium: action 可能是数组，取单个数值
        if isinstance(action, (list, tuple, np.ndarray)):
            target_pos = float(action[0])
        else:
            target_pos = float(action)

        # 限制在 [-1, 1]
        target_pos = float(np.clip(target_pos, -1.0, 1.0))

        self.prev_position = self.position
        self.position = target_pos

        idx = self.current_step
        assert idx < len(self.returns), "current_step 越界了"

        raw_ret = float(self.returns[idx])  # 基础资产日收益率

        # 组合收益（不含成本）：上一时刻持仓 * 当日收益率
        portfolio_return = self.prev_position * raw_ret

        # 换手率 = 仓位变化的绝对值
        turnover = abs(self.position - self.prev_position)
        cost = self.trading_cost * turnover

        # 风险惩罚：仓位越大惩罚越大（很简单但好解释）
        risk_penalty = self.risk_lambda * abs(self.position)

        # 奖励 = 净收益
        reward = portfolio_return - cost - risk_penalty

        # 更新权益曲线
        self.equity *= (1.0 + reward)
        self.max_equity = max(self.max_equity, self.equity)

        self.current_step += 1
        terminated = self.current_step >= len(self.returns)
        truncated = False

        obs = self._get_observation()
        info = {
            "raw_return": raw_ret,
            "portfolio_return": portfolio_return,
            "cost": cost,
            "risk_penalty": risk_penalty,
            "equity": self.equity,
            "position": self.position,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        print(
            f"step={self.current_step}, equity={self.equity:.4f}, "
            f"position={self.position:.3f}"
        )


def make_env(
    df=None,
    window_size: int = 50,
    trading_cost: float = 0.001,
    risk_lambda: float = 0.0,
):
    """
    工厂函数：创建一个自定义交易环境实例。
    """
    return CustomTradingEnv(
        df=df,
        window_size=window_size,
        trading_cost=trading_cost,
        risk_lambda=risk_lambda,
    )
