# JapaneseVowels 首轮完整运行：2026-10-13

以下是用户从 Orion 提供的 `result.json` 日志摘要与最终 `sacct` 输出；不是本地重新执行的结果。六个作业均为 `COMPLETED / 0:0`，程序均报告 `status=success, profile=benchmark`。

- 仓库 commit：`afc22f602b459dd0f6e0dd465f6769768f12b78a`。
- 数据：JapaneseVowels，官方 training 270 / test 370，12 channels，dense length 29；seed=0。Neural 方法内部划分 training/validation，保留官方 test。
- CPU：`cpup`，节点 `A386345X1309686A`；GPU：`h200p`，节点 `2FS9B74`，每个 neural 作业分配 1 张 H200。
- 每个作业分配 4 CPU、32 GiB host memory；account `prj0000000262`、QoS `highpriority`。
- 结果目录：`results/jv-first-full/benchmark/JapaneseVowels/`。
- 原始资源记录：Orion 上 `results/accounting/jv-seed0-jobs.txt`。

| Job ID | 方法 | Macro-F1 | Accuracy | fit_seconds | Elapsed 秒 | Run ID |
|---|---|---:|---:|---:|---:|---|
| 160604 | ROCKET | 0.9546861264 | 0.9513513514 | 64.715276 | 86 | 20261013T085756Z-3c8b045f |
| 160605 | BORF | 0.7114953686 | 0.7243243243 | 43.266564 | 64 | 20261013T085756Z-b8f32ad2 |
| 160606 | LGBM | 0.9331582484 | 0.9378378378 | 2.790530 | 23 | 20261013T085756Z-8fec1986 |
| 160607 | RIFC | 0.7725805933 | 0.7621621622 | 9.113225 | 38 | 20261013T085756Z-03b11803 |
| 160608 | TimesNet | 0.9606256310 | 0.9621621622 | 6.933407 | 28 | 20261013T085756Z-4ec51f53 |
| 160609 | BRITS | 0.9756672294 | 0.9756756757 | 108.378582 | 130 | 20261013T085756Z-ef1c7e99 |

TimesNet 在第 68 epoch early stop，best epoch=18；BRITS 在第 154 epoch early stop，best epoch=104。两者均符合 patience=50；1000 epochs 是上限。

## 用时与资源口径

`fit_seconds` 是整个 `.fit()` 的墙钟时间，包含该方法内部 preprocessing、training 和 validation；`Elapsed` 是 Slurm 作业占用资源的时间，不包含排队时间。两者均不是 GPU kernel time。首次 JIT 编译、导入和缓存状态会影响短任务耗时。

按主作业行计算，不重复加 `.batch`、`.extern`、`.0`：

- 四个 CPU 作业：`4 × (86+64+23+38) / 3600 = 0.23444` allocated CPU-core hours。
- 两个 GPU 作业：`(28+130) / 3600 = 0.04389` allocated H200 GPU-hours，另分配 `4 × (28+130) / 3600 = 0.17556` CPU-core hours。
- 六个作业共 `0.41000` allocated CPU-core hours。这是资源占用，不是实际 CPU 使用时间或货币价格；`billing=4/10` 也不是每小时货币单价。

BRITS 的 `160609.0` 最终 `MaxRSS=925.95M` 是 host RAM 的采样记录，不是 GPU VRAM。其他方法的 0–0.38M 明显不足以代表整个训练进程内存，原因尚未确定；本轮不据此作跨方法内存比较或降低内存申请。

## 结果的适用范围

这是一个 dataset、一个 seed，不能据此判断全部数据集排名或推算全部运行时间。BORF/LGBM 的预定 repeat=1，其余四个方法仍需 seed=1,2；全部计划 252 fits 中已完成 6，剩余 CPU 140、GPU 106。

保留 pinned upstream 的 NaN 处理。LGBM 支持 NaN；RIFC 的空区间统计可产生 NaN 并交给 LightGBM；ROCKET 的 MiniRocket transform 没有显式的 NaN-aware imputation，属于当前官方 pipeline 的限制。概率合法性与保存后重算指标通过，不代表缺失值处理语义已被全面验证。不要为了消除 warning 临时修改 pipeline 并混合结果。
