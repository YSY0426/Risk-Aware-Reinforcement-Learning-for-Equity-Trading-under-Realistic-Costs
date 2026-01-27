# configs/config.py

CONFIG = {
    # Environment settings
    "window_size": 50,
    "trading_cost": 0.001,      # 手续费系数，越大越“贵”

    # Training settings
    "total_timesteps": 50_000,  # 可以先从 50k 开始，后面再加大
    "policy": "MlpPolicy",
    "train_ratio": 0.7,

    # Paths
    "model_path": "models/ppo_custom_env.zip",
}
