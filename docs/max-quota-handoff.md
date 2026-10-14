# Orion baseline：接续到已确认的 60 任务额度

这次交付是独立 launcher，不需要 checkout 方法分支或修改正在跑的 `~/IMTS`。没有从本地连接 Orion；服务器是否已切换，以用户运行后的日志为准。

## 1. 上传并解压

把 `IMTS-max-quota-launcher.zip` 上传到 Orion 的 home 目录。压缩包内部是 `scripts/`、说明和项目记录；只需原有 baseline `.venv`，不用安装新依赖。

```bash
unzip -n "$HOME/IMTS-max-quota-launcher.zip" \
  -d "$HOME/imts-launcher-20261014"

"$HOME/IMTS/.venv/bin/python" \
  "$HOME/imts-launcher-20261014/scripts/resume_max_baselines.py" \
  --baseline-root "$HOME/IMTS"
```

默认只展示计划，不停止进程、不提交任务。不要把三个脚本复制覆盖到正在运行的 baseline repo。

## 2. 启用接续

```bash
mkdir -p "$HOME/IMTS/results/accounting"
imts_resume_log="$HOME/IMTS/results/accounting/max-quota-$(date +%Y%m%d-%H%M%S).log"

nohup "$HOME/IMTS/.venv/bin/python" -u \
  "$HOME/imts-launcher-20261014/scripts/resume_max_baselines.py" \
  --baseline-root "$HOME/IMTS" --submit --handoff \
  > "$imts_resume_log" 2>&1 < /dev/null &

printf 'Launcher PID: %s\nLog: %s\n' "$!" "$imts_resume_log"
tail -n 30 "$imts_resume_log"
```

脚本会确认 highpriority 的 MaxSubmitJobsPU=60，检查 CPU 2h、GPU 3h 能否提交，再验证旧 watcher 的 UID、路径、命令和启动时间，只停止该 watcher。已提交的 Slurm jobs 不取消，等待它们完整结束并进入 accounting 后才接续。等待可在后台完成，不需要用户等一小时后手动回来启动。

新配置是 CPU+GPU 合计最多 60 tasks，不是各 60。两个池都可使用整个额度，不为尚未运行的方法预留名额。兼容成功直接跳过；此前失败/超时重新尝试一次，不恢复训练 checkpoint。

## 3. 确认切换

```bash
pgrep -af '[s]cripts/(resume_max_baselines|submit_parallel_baselines|submit_benchmarks)\.py.*--submit'
squeue -r -u "$USER"
tail -n 40 "$imts_resume_log"
```

接续等待阶段进程名是 `resume_max_baselines.py`；开始新 watch 后变为 `submit_parallel_baselines.py`。旧日志停止增长是正常的，要看新 `max-quota-*.log`。若重新登录后变量不存在，可以 `ls -lt ~/IMTS/results/accounting/max-quota-*.log` 找到最新日志，再 tail 该完整路径。

只看见 preflight 或 waiting 不等于新任务已提交；需要看到新的 `Submitted ... array ...` 和实际 queue。60 是允许提交的任务上限，Slurm 的资源和优先级决定实际运行数，不能保证同时分配 60 块 GPU。

## 4. 成功覆盖率

```bash
cd ~/IMTS
.venv/bin/python -m imts_benchmark summarize \
  --results-dir results/jv-first-full \
  --output "results/accounting/benchmark-summary-$(date +%Y%m%d-%H%M%S).csv"
```

目标是 Coverage 108/108 combinations complete，252/252 successful fits。日志中的 deferred 不包含此前失败任务；旧 JSON 的 running 也可能是已被 Slurm 终止的遗留状态。结合 sacct、task output 与汇总定位失败，不把失败当零分。

## 限制与异常

- 仍按 wave 等整波结束，不是 rolling refill。增大提交额度解除 GPU2 瓶颈，但不保证始终占满资源。
- 用户其他 jobs 也占 60 名额。空间不足时整批停止，不自动改小或争抢额度。
- GPU 3h 比原来 1h 长，但不保证所有模型都能成功；后续根据 TIMEOUT、OOM 等实际原因处理。
- 多个 watcher、模糊 receipt、缺少 accounting、source/env 改动会拒绝接续，不绕过 ledger 检查。
- Linux/Python 不支持 pidfd 时，脚本在发信号前停止，需要另外核实后手动停止 watcher。
- preflight 失败不会停止旧 watcher。若旧 watcher 已停止后出现别的问题，已提交 Slurm jobs 仍保留；保存错误日志后处理，不盲目重复提交。

本次代码 commit message：`feat: add safe handoff to the 60-task baseline launcher`。完整重要讨论与实现记录见 `project-handoff.md`。
