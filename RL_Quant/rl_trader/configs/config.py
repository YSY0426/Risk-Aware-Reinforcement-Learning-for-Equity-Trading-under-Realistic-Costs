CONFIG = {
    # ===== Environment settings =====
    "window_size": 50,
    "trading_cost": 0.001,
    "risk_lambda": 1e-3,   # ★ 这里改成 0.0，关闭仓位风险惩罚
    "train_ratio": 0.7,

    # ===== RL training settings =====
    "total_timesteps": 500000,
    "policy": "MlpPolicy",
    "random_seed": 42,

    # ===== Paths =====
    "model_path": "models/ppo_lambda_1e-3.zip",  # ★ 建议改个新名字方便对比
}
