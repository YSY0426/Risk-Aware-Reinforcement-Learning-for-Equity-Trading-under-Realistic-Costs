# envs/market_env.py
"""
Market environment factory.

Currently we use gym-anytrading's STOCKS_GOOGL dataset and the 'stocks-v0' env.
Later you can replace the dataset and parameters according to your proposal.
"""

import gymnasium as gym
import gym_anytrading
from gym_anytrading.datasets import STOCKS_GOOGL


def make_env(window_size: int = 50):
    """
    Create a trading environment based on STOCKS_GOOGL.

    Parameters
    ----------
    window_size : int
        Number of past days included in the observation.

    Returns
    -------
    env : gym.Env
        A gym-compatible trading environment.
    """
    df = STOCKS_GOOGL
    frame_bound = (window_size, len(df))

    env = gym.make(
        "stocks-v0",
        df=df,
        frame_bound=frame_bound,
        window_size=window_size,
    )
    return env
