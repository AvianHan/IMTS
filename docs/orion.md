# Orion：从环境检查到第一次完整运行

这份文件对应六个 baseline：`rocket`、`borf`、`lgbm`、`rifc`、`timesnet`、`brits`。截至 2026-10-13，六个方法在完整 JapaneseVowels 上的 seed=0 均已在 Orion 成功，见 [首轮结果与用时](jv-first-full.md)。继续其余实验见 [批量运行说明](batch-runs.md)。全部 18 个数据集的完整 benchmark 尚未完成。

批量工具现默认每批只提交 CPU 4 / GPU 2 个 fits，以避免一次大 array 超过 QoS 的提交数量限制；`--submit --watch` 等本批结束后继续下一批。仅 `--submit` 只提交一批。具体更新、后台运行及失败记录处理见上述批量运行说明。

## 1. 在 login node 准备环境

进入服务器上的仓库目录。下面假设仓库在 `~/IMTS`，路径不同就替换这一行。

```bash
cd ~/IMTS
python3 --version
```

建议使用 Python 3.11；这组固定版本依赖适合 Python 3.10–3.12。若 Orion 必须先 `module load` 才有合适的 Python，使用学校给出的实际 module 名称；这里没有预设尚未核实的 module。

选择一种安装方式：

| 命令 | 用途 |
|---|---|
| `bash scripts/bootstrap.sh base` | 安装基础依赖；不主动安装 PyTorch |
| `bash scripts/bootstrap.sh cpu` | 安装六种方法所需依赖，PyTorch 使用 CPU wheel |
| `bash scripts/bootstrap.sh cu124` | 安装六种方法所需依赖，PyTorch 使用 CUDA 12.4 wheel |

Orion GPU 环境可先用明确选择的 CUDA 12.4 profile，随后通过计算节点上的 smoke 检查兼容性：

```bash
bash scripts/bootstrap.sh cu124
```

脚本创建仓库内 `.venv`，安装依赖，并执行 `doctor`。若默认 Python 不合适，可以明确指定，例如：

```bash
PYTHON_BIN=python3.11 bash scripts/bootstrap.sh cu124
```

PyTorch 固定为 `2.6.0`。`nvidia-smi` 显示的 CUDA 数字表示驱动能力，不能直接拿它决定 PyTorch wheel 版本。GPU 是否实际可用，以分配到的计算节点为准；login node 上看不到 GPU 不等于安装失败。

安装需要访问 Python 包源。不要在每次提交 job 时重复安装。CPU 与 CUDA profile 建议用于不同机器或单独的环境；同一个 `.venv` 切换 profile 会重新安装 PyTorch，替换已有构建。

## 2. 预先检查真实数据

```bash
.venv/bin/python -m imts_benchmark inspect --dataset JapaneseVowels
```

这一步读取数据并显示结构。首次读取可能下载数据，提前执行可减少 job 等待下载的时间。计算节点必须能读到相同的数据缓存；如果 login node 与计算节点的缓存目录不共享，应先安排数据路径再运行。

`synthetic` 是代码内生成的小样本，用来检查代码流程，不是论文数据集。先让 synthetic smoke 通过，再用 JapaneseVowels 检查真实输入。

## 3. 核实 Slurm 分区、account 和 QoS

以下是只读检查：

```bash
sinfo
sacctmgr -P show assoc where user="$USER" format=Cluster,Account,User,Partition,QOS,DefaultQOS
scontrol show partition testqueue
scontrol show partition cpup
scontrol show partition h200p
```

当前用户 `hansy` 的 account `prj0000000262` 允许的 QoS 为 `highpriority`，默认 QoS 为空。`cpun` 和 `h200n` 只接受 `normal`，与当前权限不匹配，曾导致六次提交都返回 `Invalid qos specification`；这些任务没有实际运行。

正式模板因此使用 **CPU：`cpup`；GPU：`h200p`；两者均为 `--qos=highpriority`**。以下配置已通过 Orion 的提交预检查：

```bash
sbatch --test-only --chdir="$PWD" \
  --account=prj0000000262 --partition=cpup --qos=highpriority \
  --time=02:00:00 slurm/benchmark_cpu.sbatch

sbatch --test-only --chdir="$PWD" \
  --account=prj0000000262 --partition=h200p --qos=highpriority \
  --time=01:00:00 slurm/benchmark_gpu.sbatch
```

`--test-only` 检查提交配置，不会真正提交或运行任务；其输出中的编号不表示已启动训练。`testqueue` 仍仅用于 smoke 测试。

其他用户或 account 应重新核实权限；可在提交命令上覆盖 `--account`、`--partition` 和 `--qos`。不要在未分配资源的 login node 上做 training。

## 4. 提交一个 smoke job

从仓库根目录执行：

```bash
sbatch --chdir="$PWD" slurm/smoke.sbatch
```

默认检查 **TimesNet + synthetic + CUDA**，请求 1 张 GPU、4 个 CPU core、32 GB 内存、30 分钟。先用小样本验证依赖、forward、training、prediction 和结果保存。

想检查 BRITS，就指定环境变量：

```bash
IMTS_MODEL=brits sbatch --chdir="$PWD" slurm/smoke.sbatch
```

synthetic 通过后，用真实数据做短流程检查：

```bash
IMTS_MODEL=brits IMTS_DATASET=JapaneseVowels sbatch --chdir="$PWD" slurm/smoke.sbatch
```

默认 Slurm 导出提交时的环境变量。如果站点改过该行为，可在命令中显式添加 `--export=ALL`。`--chdir="$PWD"` 让 job 从仓库根目录执行；依赖和输出路径由此保持一致。

查看队列和日志（把 `123456` 换成返回的 job ID）：

```bash
squeue -u "$USER"
tail -n 80 slurm-123456.out
sacct -j 123456 --format=JobID,State,Elapsed,ExitCode
```

`PENDING` 表示排队，`RUNNING` 表示已开始，`COMPLETED` 且 `ExitCode=0:0` 表示 job 正常退出。还应检查输出结果文件；smoke 分数只用于验证流程，不能作为 baseline 成绩。

已验证的 JapaneseVowels GPU smoke：TimesNet job `160482`、修复后的 BRITS job `160572`，两者均为 `COMPLETED`、`0:0`，日志末尾为 `"status": "success"`。它们只训练 1 个 epoch，不能替代完整训练结果。

## 5. 提交单个完整 run

CPU 方法示例：

```bash
IMTS_MODEL=rocket IMTS_DATASET=JapaneseVowels IMTS_SEED=0 \
  sbatch --chdir="$PWD" slurm/benchmark_cpu.sbatch
```

GPU 方法示例：

```bash
IMTS_MODEL=brits IMTS_DATASET=JapaneseVowels IMTS_SEED=0 \
  sbatch --chdir="$PWD" slurm/benchmark_gpu.sbatch
```

| 环境变量 | 意义 | 默认值 |
|---|---|---|
| `IMTS_MODEL` | 本次训练的方法 | CPU：`rocket`；GPU：`timesnet` |
| `IMTS_DATASET` | 本次读取的数据集 | `JapaneseVowels` |
| `IMTS_SEED` | 本次独立 run 的 seed | `0` |
| `IMTS_OUTPUT_DIR` | 结果目录 | `results`；smoke 用 `results/smoke` |
| `IMTS_MAX_DENSE_GIB` | 数据转为 dense array 时的容量上限 | `8` GiB |

CPU 模板申请 2 小时，GPU 模板先申请 1 小时；**申请时限不是预估运行时间**。超时会被 Slurm 结束，应根据第一次实际耗时和分区限制调整。完整 run 使用正式分区，smoke 使用 `testqueue`。先完成一个 dataset/model/seed，检查结果，再批量运行。

Slurm 模板把实际申请的 CPU core 数传给 CLI 的全局 `--threads` 参数，默认 4。`IMTS_MAX_DENSE_GIB` 限制数据转为 dense array 时的容量；它不代表整个 job 的内存上限，也不会替代 Slurm 的 `--mem` 请求。

## 6. 六个方法的本地短流程检查

在自己的电脑或已经获得的交互式计算资源内，可以一次检查六个方法：

```bash
bash scripts/smoke_all.sh synthetic cpu
```

已经分配到 GPU 且安装 CUDA profile 时：

```bash
bash scripts/smoke_all.sh synthetic cuda
```

此处的 `cuda` 只用于 TimesNet 和 BRITS，其他四个方法继续使用 CPU。脚本在第一个错误处停止，便于定位。它不会提交 Slurm job，不应直接在 Orion login node 执行 training。

本地脚本默认使用 1 个 CPU thread，可用 `IMTS_THREADS=4 bash scripts/smoke_all.sh synthetic cpu` 调整。该值作为全局参数放在 `smoke` 子命令之前。
