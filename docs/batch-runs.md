# 批量运行剩余 benchmark

已完成的六次 JapaneseVowels seed=0 结果与本批结果放在同一目录 `results/jv-first-full`，方便跳过已有结果和最终汇总。目录名称不限制其中可包含的数据集。

## 1. 先预览，再提交

在 Orion 的仓库根目录执行；不需要在 login node 导入 Torch 或训练模型：

```bash
cd ~/IMTS
.venv/bin/python scripts/submit_benchmarks.py
```

默认仅预览，不调用 `sbatch`。按当前进度应看到 252 个预定 fits、6 个成功记录可复用、246 个剩余 fits，其中 CPU 140、GPU 106。如果跳过数量不是 6，先检查当前结果目录和脚本输出，不要通过移动/删除成功记录来强行凑数。

确认清单符合预期后，执行真正提交命令：

```bash
.venv/bin/python scripts/submit_benchmarks.py --submit
squeue -u "$USER"
```

`--submit` 才会提交。脚本使用两组 Slurm job arrays：CPU 默认最多 4 个作业同时运行，GPU 默认最多 2 个作业同时运行。每个作业只执行一个 dataset/model/seed，互不共享训练状态；一个失败不会使其他 array task 自动失败。Slurm 或账号限制可能让实际并发更低。

| 组别 | 方法 | 每个作业的资源 | 默认时限 | 并发上限 |
|---|---|---|---|---|
| CPU | ROCKET、BORF、LGBM、RIFC | 4 CPU、32 GiB RAM，cpup/highpriority | 2 小时 | 4 |
| GPU | TimesNet、BRITS | 4 CPU、32 GiB RAM、1 GPU，h200p/highpriority | 1 小时 | 2 |

这些是初始资源预算，不是对其余数据集耗时和内存的预测。dense load 仍默认限制为 8 GiB；不会缩短序列或抽样来绕过容量限制。某些大数据集可能需要更高内存或时限，失败后按日志分别处理。

可以减少并发，例如：

```bash
.venv/bin/python scripts/submit_benchmarks.py --submit \
  --cpu-concurrency 2 --gpu-concurrency 1
```

也可以只选择部分数据集。以下默认仍只是预览：

```bash
.venv/bin/python scripts/submit_benchmarks.py --datasets JapaneseVowels
```

JapaneseVowels 此时应剩余 8 个 fits：ROCKET、RIFC、TimesNet、BRITS 各 seed=1,2。

## 2. 为什么已有结果能继续用

脚本读取两个现有 JSON 配置，按规定的数据集顺序、方法和 seeds 生成清单。只有 `profile=benchmark` 且 `status=success`、来源和环境检查匹配的记录才算已完成。smoke 和失败记录不计分；重复成功、代码/依赖不一致会报错。

本次新增内容没有修改 `src/` 或 `configs/`，原有 `source_sha256` 不变。新的 git commit 只记录提交脚本和说明的变更；不要在这批任务尚未结束时修改训练代码、配置或升级依赖。worker 开始时还会检查计划中的代码和环境。

每批提交在 `results/accounting/batches/<batch-id>/` 保存 `manifest.json` 清单、脚本快照以及 CPU/GPU 的 intent 和 receipt（提交意图与回执）。有仍在排队/运行的相关批次或无法核实的提交记录时，脚本阻止重复提交。不要删除这些记录或绕过脚本手动提交同一批 fits。并发上限针对当前 array，不是整个账号的所有作业。

## 3. 看哪里判断结果

- `squeue -u "$USER"`：看排队和正在执行的 tasks；array 中可能出现 `123456_[0-139%4]` 这样的合并表示。
- `results/accounting/batches/<batch-id>/slurm-<array-id>_<task-index>.out`：每个 array task 有单独日志，`%A` 对应 array job ID，`%a` 对应 task index。
- `sacct -j <CPU_ARRAY_ID>,<GPU_ARRAY_ID> --format=JobID,State,Elapsed,ExitCode`：看结束状态。每个 task 要结合 `COMPLETED / 0:0` 与其日志末尾的 `status=success` 判断。
- `results/jv-first-full/benchmark/<dataset>/<MODEL>/seed-<seed>/<run-id>/result.json`：本次分数、fit/predict/total 用时、代码与依赖记录。

失败不在同一批里自动重试。先检查 `TIMEOUT`、`OUT_OF_MEMORY`、下载错误或 Python traceback；确定原因后，再显式提交所需数据集的剩余任务。进程被 Slurm 强制结束时，`result.json` 可能留在 `running`，所以不能只看 JSON 状态判断任务是否还活着。由此脚本管理的旧 array 在 Slurm 确认全部结束后可续跑；没有对应提交记录的旧 `running` 文件会要求人工核实，避免误重跑仍活跃的手动任务。

`--cpu-time`、`--gpu-time` 可显式修改下一批的时限；`--max-dense-gib` 可修改 dense load 的保护阈值。但提高后者之前必须同时确认 Slurm 内存申请足够：它不是增加 RAM 的选项。32 GiB RAM 来自对应 `slurm/benchmark_*.sbatch` 模板，不应只为绕过错误就盲目提高容量阈值。

## 4. 下载和成本的口径

worker 在模型运行前，以同一数据文件的跨进程锁保护 cache fetch，避免多个作业同时重复下载同一个文件；使用官方 pinned registry 校验。它不在这一步加载 dense array，也不改变训练输入。

首次下载和等待 cache 锁计入 Slurm `Elapsed`，不计入 runner 的 `total_seconds` 或 `fit_seconds`。比较完整成本时保留 Slurm 记录；比较 `.fit()` 耗时时仍使用 `fit_seconds`。本批 Slurm 时长可能包含首次数据下载，不能直接当作纯训练性能。

所有 tasks 结束后，用两组真实 array ID 保存资源记录，保留 step rows；计算总资源时只加主 task，不能把 `.batch`、`.extern`、`.0` 重复相加：

```bash
mkdir -p results/accounting
sacct -j <CPU_ARRAY_ID>,<GPU_ARRAY_ID> --units=M \
  --format=JobID,JobName,Partition,NodeList,State,Elapsed,AllocCPUS,AllocTRES,TotalCPU,MaxRSS,ExitCode \
  -P > results/accounting/batch-jobs.txt
```

`<CPU_ARRAY_ID>`、`<GPU_ARRAY_ID>` 是待替换的占位符，不能原样执行。若分多批运行，资源记录也应使用不同文件名，保留每一批。

## 5. 最终汇总

任务结束后执行（输出 CSV 已存在时换一个新文件名）：

```bash
.venv/bin/python -m imts_benchmark summarize \
  --results-dir results/jv-first-full \
  --output results/accounting/benchmark-summary.csv
```

目标覆盖率是 **108/108 dataset/model combinations、252/252 successful fits**。只有预定 seeds 全齐的组合才是 `complete=True`；失败或缺失不作为零分计入均值。

Slurm 官方说明：[job arrays 与并发限制](https://slurm.schedmd.com/job_array.html)、[sbatch](https://slurm.schedmd.com/sbatch.html)、[sacct](https://slurm.schedmd.com/sacct.html)。
