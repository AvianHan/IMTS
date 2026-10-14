# Orion 今晚提交方法实验

本轮在 `feature/our-method` 中运行 10 个历史、写入、交互配置。先 30 次 validation 检查，全部成功后自动扩展为 18 datasets × 10 配置 × 3 seeds = 540 次正式实验。详细含义见 [method-ablations.md](method-ablations.md)。

## 1. 保留两个独立目录

正在运行的 baseline 留在 `~/IMTS`；方法使用 `~/IMTS-method`、独立 `.venv` 和结果目录。无需停止、替换或升级当前 baseline watcher，也不要在 baseline 目录切 branch。

先取得交付的 `IMTS-our-method.bundle`。这是 Git branch 的离线传输文件，不是额外 launcher；这里没有 GitHub 写入凭据，所以没有代替用户 push。把它上传为 Orion 的 `~/IMTS-our-method.bundle`，然后首次执行：

```bash
cd ~/IMTS
git bundle verify ~/IMTS-our-method.bundle
git fetch ~/IMTS-our-method.bundle feature/our-method:feature/our-method
git worktree add ~/IMTS-method feature/our-method
cd ~/IMTS-method
bash scripts/bootstrap.sh cu124
```

如果目录/branch 已经存在，先 `git -C ~/IMTS worktree list` 和 `git -C ~/IMTS-method status --short --branch`，只在确认方法目录没有未提交修改后导入更新；不要强制覆盖。bundle 在已存在的个人电脑 repo 中也可 `git fetch`，再 `git push origin feature/our-method`。推送成功后，Orion 可改用 `git fetch https://github.com/AvianHan/IMTS.git feature/our-method` 获取同一提交。

运行期间保持方法源码、configs 和 Python 环境不变。原始数据下载缓存可以复用；预处理和训练使用方法自己的代码与环境。

## 2. 先看计划和 Slurm 接受情况

在与当前 baseline watcher 相同的 login node 上运行（当前为 `orion-login01`），这样能识别它的本地进程并协调名额。

```bash
cd ~/IMTS-method
.venv/bin/python scripts/submit_method_pilot.py \
  --group ablation --staged --evaluate-test --watch \
  --baseline-root ~/IMTS --output-dir results/method-ablation-full

.venv/bin/python scripts/submit_method_pilot.py \
  --group ablation --staged --evaluate-test --watch \
  --baseline-root ~/IMTS --output-dir results/method-ablation-full \
  --test-only
```

第一条是 dry run，不训练、不提交；应列出 30 次检查和 540 次正式 fit。第二条是 `sbatch --test-only`，不预留资源。失败时看具体错误，先不要执行下一步。

## 3. 一次启动 分阶段自动继续

```bash
cd ~/IMTS-method
mkdir -p results/accounting
imts_method_log="results/accounting/method-watch-$(date +%Y%m%d-%H%M%S).log"
nohup .venv/bin/python -u scripts/submit_method_pilot.py \
  --group ablation --staged --evaluate-test \
  --baseline-root ~/IMTS --output-dir results/method-ablation-full \
  --submit --watch > "$imts_method_log" 2>&1 < /dev/null &
printf 'Watcher PID: %s\nLog: %s\n' "$!" "$imts_method_log"
tail -n 25 "$imts_method_log"
squeue -r -u "$USER"
```

每个 task 申请 1 GPU、4 CPU cores、32 GiB、3 小时。最长时间不是训练耗时保证，超过时限仍会失败；不会静默缩短 epochs 或从半截 checkpoint 假装续训。

**没有每批 4 个的限制。** 根据已确认的 QoS，总 running+pending 上限 60；完成任务后每 30 秒检查并补位，不等整批结束。检查阶段本来只有 30 个任务，正式阶段才有 540 个待执行项。其他个人作业也计入总额。集群空闲资源与优先级决定实际同时运行多少。

当前旧 baseline 仍需两 GPU task 名额。方法会识别它并协调这两个名额，防止它下一波提交被挤到 QoS 上限；baseline watcher 退出后自动释放。不是长期给未来任务空留四个名额。不要在不同 login node 重复启动 watcher；不同主机看不到彼此的 `/proc`，需手动 `--reserve-other-jobs 2` 协调旧 watcher。

30 次检查任一失败，脚本停止扩大到正式阶段，并留下日志。检查通过会打印 `Stage gate OPEN` 并记录证据。只有 checkpoint、预测、诊断及来源均通过检查的成功才复用。模糊提交回执不会盲目重试；明确失败需检查后用 `--retry-failed` 重启同一计划，每轮只尝试一次。

## 4. 关电脑和明早看结果

`sbatch` 成功后任务归 Slurm 管；个人电脑关机、离开公司 Wi-Fi 或 SSH 断线不取消它。上述 `nohup ... &` 让服务器上的 watcher 脱离普通 SSH 挂断，断线后仍能补交后续任务。登录节点重启或管理员清理进程仍可能结束 watcher；已经提交的 Slurm jobs 独立继续。不要直接在 SSH 前台跑完整训练后就关机。

重新登录后：

```bash
cd ~/IMTS-method
pgrep -af '[s]cripts/submit_method_pilot.py.*--submit.*--watch'
squeue -r -u "$USER"
ls -t results/accounting/method-watch-*.log
```

用 `tail -n 35` 查看上面对应日志文件。正式阶段应看到 `test: successful 540/540`。队列空或 watcher 结束本身不代表成功。

汇总和 baseline 对照：

```bash
cd ~/IMTS-method
.venv/bin/python -m imts_agents summary --output-dir results/method-ablation-full
.venv/bin/python -m imts_agents compare \
  --method-root results/method-ablation-full \
  --baseline-root ~/IMTS/results/jv-first-full \
  --output "results/accounting/comparison-$(date +%Y%m%d-%H%M%S).csv"
```

baseline 单独进度仍用原命令：

```bash
cd ~/IMTS
.venv/bin/python -m imts_benchmark summarize \
  --results-dir results/jv-first-full \
  --output "results/accounting/benchmark-summary-$(date +%Y%m%d-%H%M%S).csv"
```

目标为方法 540/540 official test fits、baseline 252/252 fits。缺失/失败单独报告。明早能看到多少取决于排队、训练耗时和失败情况，不能承诺一晚全部完成。
