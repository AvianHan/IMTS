# 本轮实现验证

范围：feature/our-method 的 10 个 history/write/interaction 配置，以及现有提交器的分阶段滚动补位。这里记录本地验证；没有在 Orion 或 H200 上代替用户提交任务。

## 自动检查

完整 pytest：**282 passed**。包括 train-only 数据边界、state-only 读取、冻结 ST memory、gate 梯度与初始保留、路由/COMMIT、validation/test 隔离、诊断、结果汇总，以及 Slurm mock 下的 quota、rolling refill、stage gate、精确复用、失败和重复提交控制。

独立审查确认：没有旧 raw token 绕过 state-only 的直接读取；history/write/interaction 没有未解决的阻塞项。旧 12 variants 的 seeded parameters 和已有输出与此前提交逐项一致。

## 真实数据 smoke

| Dataset | 配置数 | Seed | Epochs | 结果 |
|---|---:|---:|---:|---|
| JapaneseVowels | 10 | 0 | 1 | 10/10 success |
| CharacterTrajectories | 10 | 0 | 1 | 10/10 success |
| Physionet2012 | 10 | 0 | 1 | 10/10 success |

全部使用 CPU、1 thread、完整 train/validation split。每项 checkpoint、预测重建、有限指标和动作覆盖检查通过；没有 official test scoring。这是 plumbing readiness，不是收敛证明或与 baseline 的成绩比较。不能据一轮 smoke 的分数判断哪个机制更好。

运行证据：`references/method-ablation-smoke-summary.json`；完整原始输出在交付 zip 的 evidence 中。

这 30 次的 source SHA256 为 `8d5d0aadf07dc651f9b66e215b825c434f6439c3bf840327937cd3c5c8ad7777`。之后仅修补两个收尾问题：释放 xarray accessor 的引用循环，以及汇总中失败/运行记录计数；model 和 runner 没再改变。最终源码另对 Physionet2012/history_state 做一次 smoke，结果记录在 `references/method-final-check.json`，不会把两个源码版本混入正式比较。

## 数据内存

预处理改为逐样本清洗/标准化，train-only feature moments 使用有界工作区；float64 缩放原地执行，稀疏 frame 在 patch 分配之前释放并显式回收 accessor cycle。保留 16GiB preparation guard，并记录估算范围。不是直接取消保护。

Physionet2012 预处理实测 peak RSS 从约1244MiB降至818MiB（显式 cycle 回收之前的测量）。JapaneseVowels、CharacterTrajectories、Physionet2012 的数据准备与训练均已实跑。

固定 revision 的 InsectWingbeat 元数据实际为 50000 samples、200 channels、22 timeline positions、67106800 stored records。managed preparation peak 估算约12.94GiB，低于16GiB guard及32GiB job请求。只读取了320KiB HDF5元数据，**未执行完整 InsectWingbeat 准备/训练**；上游稀疏读取另有瞬时峰值，估算不是全进程RSS保证。证据在 `references/method-data-readiness.json`。

## 提交与结果边界

离线 dry run 确认30pilot+540full=570fits，无4task限制。mock测试覆盖全用户60cap、旧baseline两名额的协调、其他个人作业占用、array部分完成后立即补位、重复worker锁、未知receipt停止，以及pilot失败不扩大全量。

CUDA训练、真实Slurm政策和资源可用性需由Orion预检查和首阶段验证。任何pilot失败都会关住全量阶段。完成判断以 `METHOD PLAN COMPLETE: 540/540 verified successful fits` 及正式结果覆盖率为准。

写入 gate 统计的极小非零std可能来自float32归约误差；不能把近零数值误差解释成static gate随输入变化。soft gate和role-update计数均不能直接当成节省FLOPs的证据。
