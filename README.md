# RL-Quant 实验运行步骤说明文档

## 1. 项目目标

本项目的目标是基于 **PPO（Proximal Policy Optimization）** 和自定义交易环境，完成一个可复现的深度强化学习量化交易实验流程，并通过与 **Buy and Hold** 基线策略对比，分析不同奖励函数参数（尤其是 `risk_lambda`）对模型表现的影响。

---

## 2. 每次实验前要做的准备

### 2.1 打开 Anaconda Prompt

先进入你配置好的 Conda 环境，并进入项目目录：

```bash
conda activate rl-trader
D:
cd D:\university\senior\FYP\RL_Quant\rl_trader
```

确认终端前缀显示为：

```bash
(rl-trader)
```

这表示你已经进入正确的 Python 虚拟环境。

---

## 3. 修改实验参数

实验参数统一写在：

```bash
configs/config.py
```

打开方式：

```bash
notepad configs\config.py
```

当前推荐配置模板如下：

```python
CONFIG = {
    # ===== Environment settings =====
    "window_size": 50,
    "trading_cost": 0.001,
    "risk_lambda": 0.0,
    "train_ratio": 0.7,

    # ===== RL training settings =====
    "total_timesteps": 50_000,
    "policy": "MlpPolicy",
    "random_seed": 42,

    # ===== Paths =====
    "model_path": "models/ppo_custom_env_googl_norisk.zip",
}
```

### 3.1 各参数含义

- `window_size`：状态中使用多少天的历史收益作为输入。
- `trading_cost`：交易成本系数，按换手率收取。
- `risk_lambda`：风险惩罚系数，用于控制仓位惩罚强度。
- `train_ratio`：训练集占全部数据的比例，其余用于测试。
- `total_timesteps`：PPO 的训练总步数。
- `policy`：策略网络类型，目前使用 `MlpPolicy`。
- `random_seed`：随机种子，便于实验复现。
- `model_path`：模型保存路径。

### 3.2 实验建议

每次修改关键参数（尤其是 `risk_lambda`、`trading_cost`）时，建议同步修改 `model_path`，以免覆盖之前训练好的模型。

例如：

- `risk_lambda = 0.001`
  - `model_path = "models/ppo_custom_env_risk001.zip"`
- `risk_lambda = 0.0`
  - `model_path = "models/ppo_custom_env_norisk.zip"`
- `risk_lambda = 0.002`
  - `model_path = "models/ppo_custom_env_risk002.zip"`

---

## 4. 训练模型

参数修改完成后，运行训练脚本：

```bash
python train.py
```

### 4.1 训练脚本会做什么

`train.py` 的主要流程是：

1. 读取 `config.py` 中的参数。
2. 加载 `STOCKS_GOOGL` 数据集。
3. 按 `train_ratio` 将数据切分为训练集和测试集。
4. 在训练集上创建 `CustomTradingEnv`。
5. 使用 PPO 算法进行训练。
6. 将训练完成的模型保存到 `model_path` 指定位置。

### 4.2 成功训练的标志

训练成功后，终端会出现类似输出：

```text
[INFO] Saving model to models/xxx.zip ...
[INFO] Training finished successfully!
```

---

## 5. 在测试集上回测 PPO

训练完成后，运行：

```bash
python backtest.py
```

### 5.1 回测脚本会做什么

`backtest.py` 的主要流程是：

1. 读取配置参数。
2. 使用同样的参数创建测试环境。
3. 将后 30% 数据作为测试集。
4. 加载训练好的 PPO 模型。
5. 在测试集上运行完整 episode。
6. 输出评价指标并绘制权益曲线。

### 5.2 重点关注的输出指标

```text
[RESULT] Final equity: ...
[RESULT] Total net return (approx): ...
[RESULT] Annualized return (approx): ...
[RESULT] Annualized volatility (approx): ...
[RESULT] Sharpe ratio (approx): ...
[RESULT] Max drawdown (approx): ...
```

这些指标分别表示：

- **Final equity**：测试结束时的资金净值。
- **Total net return**：测试区间总收益率。
- **Annualized return**：年化收益率。
- **Annualized volatility**：年化波动率。
- **Sharpe ratio**：风险调整后收益指标。
- **Max drawdown**：最大回撤。

---

## 6. 运行 Buy and Hold 基线策略

为了和 PPO 做对比，需要运行简单基线策略：

```bash
python baseline_buy_and_hold.py
```

### 6.1 基线脚本会做什么

`baseline_buy_and_hold.py` 使用与 PPO 相同的测试集，假设从测试开始时满仓持有到结束，输出同样的一组指标。

### 6.2 重点关注输出

```text
[BASELINE] Final equity: ...
[BASELINE] Total net return: ...
[BASELINE] Annualized return: ...
[BASELINE] Annualized volatility: ...
[BASELINE] Sharpe ratio: ...
[BASELINE] Max drawdown: ...
```

---

## 7. 标准实验流程

以后每次做实验都按以下顺序进行：

### Step 1：进入环境和目录

```bash
conda activate rl-trader
D:
cd D:\university\senior\FYP\RL_Quant\rl_trader
```

### Step 2：修改配置文件

```bash
notepad configs\config.py
```

### Step 3：训练 PPO

```bash
python train.py
```

### Step 4：测试 PPO

```bash
python backtest.py
```

### Step 5：测试 Buy and Hold baseline

```bash
python baseline_buy_and_hold.py
```

### Step 6：记录实验结果

将 PPO 和 baseline 的结果整理到实验表格中。

---

## 8. 实验结果记录模板

建议每次实验都整理成下表：

| Experiment | risk_lambda | trading_cost | Final Equity | Total Return | Ann. Return | Ann. Vol | Sharpe | MaxDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 0.001 | 0.001 | 0.9317 | -6.83% | -2.70% | 16.68% | -0.08 | -20.76% |
| PPO | 0.0 | 0.001 | 1.6261 | +62.61% | +20.74% | 18.33% | 1.12 | -14.83% |
| PPO | 0.002 | 0.001 | 0.6237 | -37.63% | -16.72% | 14.21% | -1.22 | -40.81% |
| Buy & Hold | — | — | 1.6953 | +69.53% | +20.93% | 20.13% | 1.05 | -15.36% |

---

## 9. 目前已完成的关键实验

### 9.1 默认环境 `stocks-v0`

- PPO 学到“do nothing”策略；
- reward 基本为 0；
- equity 曲线接近一条平线；
- 主要作用：验证训练—回测 pipeline 已经跑通。

### 9.2 自定义环境 + `risk_lambda = 0.001`

- 加入固定仓位风险惩罚；
- 样本外测试表现较差；
- 说明过强的风险惩罚会导致模型过于保守。

### 9.3 自定义环境 + `risk_lambda = 0.0`

- 去掉固定风险惩罚；
- PPO 表现显著提升；
- 样本外收益接近 Buy and Hold，Sharpe 略高。

### 9.4 自定义环境 + `risk_lambda = 0.002`

- 加重风险惩罚；
- PPO 表现明显恶化；
- 说明固定仓位惩罚过大会显著压制收益。

---

## 10. 当前阶段的主要结论

根据目前实验，可以总结为：

1. **默认环境不适合作为最终研究环境**，因为 PPO 容易学成“零交易策略”。
2. **自定义环境是有效的**，它可以产生非零 reward、真实权益曲线和可解释的风险收益特征。
3. **固定的仓位风险惩罚并不一定带来更好的风险收益表现**：
   - `risk_lambda` 越大，模型越保守；
   - 在单资产上涨行情里，过强惩罚反而使 PPO 明显落后于 Buy and Hold。
4. **在 `risk_lambda = 0` 时，PPO 已经能接近 Buy and Hold 的收益，同时在 Sharpe 和 Max Drawdown 上略有优势。**

---

## 11. 下一步建议

建议继续尝试以下实验：

1. **中间值实验**：
   - `risk_lambda = 0.0002`
   - `risk_lambda = 0.0005`

2. **交易成本敏感性实验**：
   - `trading_cost = 0.0005`
   - `trading_cost = 0.002`

3. **窗口长度实验**：
   - `window_size = 100`

4. **训练步数实验**：
   - `total_timesteps = 100000`
   - `total_timesteps = 200000`

每次实验都建议只改变一个变量，其他参数保持不变，这样才能分析具体是什么因素导致了表现变化。

---



