# 已执行验证记录

验证日期：2026-10-13。环境：Linux / Python 3.12 / PyTorch 2.6.0+cpu。

**下文记录本地代码与短流程验证。** Orion 上六个方法的完整 JapaneseVowels seed=0 已通过，结果与资源记录见 [首轮完整运行](jv-first-full.md)，集群操作见 [Orion 说明](orion.md)。全部 18 个数据集的完整 benchmark 尚未完成。

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

## 批量提交工具验证

新增 `tests/test_batch_submission.py` 使用临时结果目录和模拟 Slurm 回应，验证 252→246（CPU 140 / GPU 106）的跳过逻辑、dry-run 不提交也不写文件、来源/环境不匹配与重复成功拒绝、活跃/未知提交阻止重试、完整 array accounting 核实、部分提交保留回执、生成脚本的 `bash -n` 检查、worker 代码变更检查和任务锁。

这部分是离线提交逻辑测试，不代表已经在 Orion 执行新的 arrays，也不产生新的 benchmark 分数。模型和数据加载代码没有因批量提交而改变。
