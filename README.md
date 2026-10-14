# IMTS

Pyrregular multivariate classification baseline 复现：**ROCKET、BORF、LGBM、RIFC、TimesNet、BRITS**。

当前阶段是建立可检查的 baseline 实验流程。完整 benchmark 尚未运行；`smoke` 分数不代表论文复现结果。

研究方法的独立实现见 [方法与消融总览](docs/method-overview.md)，首轮实际验证见 [JapaneseVowels 方法 pilot](docs/method-pilot-results.md)，与 baseline 并行运行见 [方法分支及 Orion 操作](docs/method-orion.md)。四组主实验分别开关「跨 time patch 的 recurrent memory」和「内层 T/S/ST 关系分工」，另有八组动机消融。默认只评估 validation；它们属于研究原型，不是正式 baseline 成绩。

## 先看结构

| 文件 | 作用 |
|---|---|
| `configs/datasets.json` | 按论文顺序列出 18 个 multivariate 数据集、channels、数据文件 SHA-256 |
| `configs/baselines.json` | 方法、重复 seeds、源码与数据版本 |
| `src/imts_benchmark/data.py` | 加载官方数据，保留原始 train/test split 和 NaN |
| `src/imts_benchmark/models.py` | 建立六个官方 baseline 的新实例并设置 seed |
| `src/imts_benchmark/compat.py` | BRITS validation 数据源的一处已记录修复 |
| `src/imts_benchmark/runner.py` | fitting → prediction → metrics → 保存与重新读取检查 |
| `src/imts_benchmark/metrics.py` | 显式计算 macro-F1、accuracy、macro precision/recall |
| `slurm/` | Orion 的 smoke / CPU / GPU 提交模板 |
| `references/` | 官方参考 F1 表、源码校验摘要 |
| `tests/` | 数据隔离、metric、BRITS validation 和结果汇总检查 |

## 安装与验证

本项目使用 editable installation，请保留完整目录结构。Python **3.10–3.12**；本次本地验证使用 3.12 / CPU。服务器 GPU 验证须在计算节点执行。

```bash
# CPU 环境，包含全部六种方法；约需几分钟下载依赖。
bash scripts/bootstrap.sh cpu

# 检查环境与数据；数据下载后缓存。
.venv/bin/python -m imts_benchmark doctor
.venv/bin/python -m imts_benchmark inspect --dataset JapaneseVowels

# 测试代码；BRITS 测试会核对 validation loader 实际读取 held-out 数据。
.venv/bin/python -m pytest -q

# 六种方法的合成数据短流程，随后检查真实数据接口。
bash scripts/smoke_all.sh synthetic cpu results/verification
bash scripts/smoke_all.sh JapaneseVowels cpu results/verification
```

Orion 上不要在 login node 运行 training。GPU 环境使用 `bash scripts/bootstrap.sh cu124`，并按 [Orion 操作步骤](docs/orion.md) 提交 `testqueue` smoke。CPU 环境和 CUDA 环境分别记录实际版本；不能根据 `nvidia-smi` 显示的最高 CUDA 版本推断应该安装哪一种 wheel。

## 一次完整实验

```bash
# 在本地 CPU 或已分配的计算节点执行；不在 login node 执行。
.venv/bin/python -m imts_benchmark --threads 4 run \
  --dataset JapaneseVowels --model ROCKET --seed 0 --output-dir results/experiment-001
```

`run` 不缩减样本或时间长度，不降低 epoch 上限。BRITS/TimesNet 使用官方 1000 epoch 上限、patience=50，实际可能 early stop。`smoke` 保留 model 结构，将 neural training 限为 1 epoch，并只在原 train/test 各自内部取小样本；其结果放在独立目录。

每次 run 保存：`result.json`（参数、环境、状态、时间、metric）、`split_ids.json`（样本 ID）、`predictions.npz`（逐样本预测与概率）、`training.log`。失败时还有 `error.txt`；已完成结果不会被覆盖。PyPOTS 等库的 stderr 同时保留在终端或 Slurm log。当前不保存供后续部署的 fitted model weights。

```bash
# 18 数据集 × (3+1+1+3+3+3 次) = 252 个完整 fits；仅生成计划，不提交。
.venv/bin/python -m imts_benchmark plan --output experiment-plan.csv

# 只汇总 benchmark success；排除 smoke/failed，并拒绝混合版本或重复 seed。
.venv/bin/python -m imts_benchmark summarize \
  --results-dir results/experiment-001 --output benchmark-summary.csv
```

`complete=False` 表示该 dataset/model 的预定 seeds 没跑齐。缺失论文参考值保持空白；不能当作 0。`macro_f1_std` 使用 sample standard deviation (`ddof=1`)，论文历史统计实现未公开。这里不声称与作者的所有小数逐位一致。

## 复现边界

- 官方源码固定到 `94644559dbe0d92bc0a10105c8d2ce300b700c97`，数据固定到 `f416b1b432f248a73323222f2378d0e8433ca6d2`。
- **ROCKET** 对应 MiniRocketMultivariateVariable → LightGBM；**BORF** 使用论文的 aeon 实现。
- 不额外加 normalization、imputation 或 timestamp channel。BRITS 在这套官方接口中没有接收真实 timestamps，其 delta 来自离散位置上的 missing mask。
- BRITS/TimesNet 从官方 fitting split 内再留出 25% validation；test 不参与 training 或 early stopping。
- ROCKET/RIFC/TimesNet/BRITS 各用 seeds `0,1,2`；BORF/LGBM 各用 seed `0`。这些是我们的可复现 seeds，并非已找回的作者原始 seeds。
- 作者没有提供完整历史依赖 lock。因此 `requirements*.txt` 是本项目固定并验证的环境，不冒充作者原环境。
- **PyPOTS 1.5 的 BRITS validation 数据源有 bug**。本项目局部 subclass 仅把 validation dataset 的 `train_set` 改为 `val_set`，不修改安装包；结果会记录该修复。详见 [protocol](docs/protocol.md)。

## 参考

- [Pyrregular 官方仓库](https://github.com/fspinna/pyrregular)
- [ICLR 2026 论文](https://openreview.net/forum?id=qetBM8nLkf)
- [本项目的实验协议与差异](docs/protocol.md)
- [已执行验证记录](docs/validation.md)
