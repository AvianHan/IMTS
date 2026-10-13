# 已执行验证记录

验证日期：2026-10-13。环境：Linux / Python 3.12 / PyTorch 2.6.0+cpu。

**这是代码与短流程验证，不是完整 benchmark 复现成绩。** 下文记录本地验证；Orion CUDA/H200 上的 TimesNet、BRITS smoke 已通过，当前集群进展见 `docs/orion.md`。全部 18 个数据集的完整 benchmark 尚未完成。

## 结果

| 检查 | 结果 |
|---|---|
| Unit/regression tests | 31 passed |
| 官方源码 hash / 数据 revision | 通过 |
| JapaneseVowels 完整数据加载 | train (270,12,29), test (370,12,29) |
| synthetic 六方法 fit → predict_proba → 保存 → 重读 | 全部通过；train40 / test16 |
| JapaneseVowels 六方法短流程 | 全部通过；原split内抽样 train64 / test32 |
| BRITS validation loader | 确认读取 held-out samples，不是 training samples |
| PyPOTS 第二epoch故意失败 | 已有checkpoint仍被标为失败，不进入成功结果 |
| Slurm/bash 语法 | 五个脚本 bash -n 通过 |

六方法为 ROCKET、BORF、LGBM、RIFC、TimesNet、BRITS。neural smoke 保留正式结构，仅 epochs=patience=1。所有 smoke 均不进入论文结果汇总。

机器可读的实际执行记录在 `references/smoke_evidence.json`。最后的 BRITS 真实数据 smoke 还验证了 source hash 检查、validation IDs 保存与 failure monitor 的组合。

sktime 对含NaN的 pipeline 给出 capability warning；本项目保留官方输入和实现，不借此自行加 imputation。预测概率和落盘后的分数均通过验证，但这些小测试不能保证每个大型数据集都不会出错。

精确 CPU 验证依赖见 `requirements-lock-cpu-py312.txt`；常规安装设置见 `requirements.txt` / `requirements-neural.txt`。作者历史依赖未公开。
