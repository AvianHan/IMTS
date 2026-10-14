# 将交付包放进同一个 GitHub repo 的方法分支

交付包包含 `changes.patch`、一份代码快照 `repository/`、首轮完整审计输出 `pilot-results/`。补丁对应你已安装 bounded-watch 更新的 baseline 工作副本；若后续又改过相同行，先看 `git apply --check` 的结果，不要强行覆盖。

在自己电脑上的 IMTS repo 操作。先确认 `git status --short` 没有待处理的改动，再从最新 main 建立方法分支。下面把解压目录记作 `/path/to/IMTS-method-delivery`，替换成实际路径：

```bash
cd /path/to/IMTS
git status --short
git fetch origin
git switch -c feature/our-method origin/main
git apply --check /path/to/IMTS-method-delivery/changes.patch
git apply /path/to/IMTS-method-delivery/changes.patch
git diff --check
git status --short
git add README.md docs references scripts slurm src/imts_agents tests
git commit -m "feat: add recurrent T/S/ST method, ablations, and quota-aware pilots"
git push -u origin feature/our-method
```

如果该分支已经存在，先查看其内容再切换；不使用强制重置。补丁检查若报错，保留报错和 `git status`，根据差异合并，不直接用快照覆盖整个 repo。

`pilot-results/` 是本地 CPU、JapaneseVowels seed 0 的探索性 validation 实验，包含锁定 checkpoint、预测、训练 history 和 provenance；不要将它复制到 Orion baseline 结果目录充当正式结果。紧凑摘要已在补丁的 `references/method-pilot-summary.json` 中。

推送成功后，按 [Orion 独立 worktree 与并行运行步骤](method-orion.md) 操作。只在新的 `~/IMTS-method` 创建方法环境，现有 `~/IMTS` 继续作为 baseline 训练目录。

合并时可使用正常 PR 合并。方法代码位于新 package `src/imts_agents`，与 baseline package 分开；共享 launcher 的修改仍需正常 review。等原 baseline 任务和 watcher 完成后再让其工作目录更新到含新源码的 commit，否则现有任务的源码指纹检查会拒绝执行。
