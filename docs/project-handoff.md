# IMTS 研究与实验接续记录

更新于 2026-10-14 本轮实现。**以本文和 method-ablations.md 为当前决定；旧讨论完整保存在 project-handoff-before-ablations.md。** 不要将准备好的本地代码说成已经部署 Orion。

## 用户当前决定

- 在同一 repo 的 `feature/our-method` 开发，Orion 使用独立 `~/IMTS-method`；正在运行的 baseline 留在 `~/IMTS`。
- 加入三个问题的 ablation：历史怎样保留、新信息应该写给谁、角色怎样交互。
- 保留 T/S/ST，不再做“为什么分 T/S/ST”的对照。用户以 TimeFilter 为设计参考；不要重复要求角色数/ALL 相关实验。
- 先检查，没问题后扩展到全部 **18 datasets**，与之前六个 baseline 一起比较。JapaneseVowels 仅是检查集之一。
- 当前旧 baseline watcher 保持运行，本轮不交接、不替换、不停止。后续新方法提交去掉每批 4 个和整波等待，使用确认的 CPU+GPU combined 60 个 running+pending 名额。
- 用户今晚自行在 Orion 提交。此工作环境没有服务器 SSH 会话，也没有 GitHub 写入凭据；不能声称已经提交 server/push。

## 沟通约定

中文解释，保留必要 English technical terms。用户基础需要循序渐进：一次一个机制、先例子、再位置、最后表格或公式。此前 overview 太密导致难理解。重要讨论、实现和实验状态持续记录，方便新 chat 接续。

交付代码必须给 commit message，并说清 commit、push、server deployment 三者。可直接用 terminal 的事情不要额外发明 launcher。已经确认的额度不要反复让用户重查。最大提交额度不等于专属 GPU，也不能保证明早全完成。

## 当前方法和实验范围

方法先把 irregular records 编码为 8 个 chronological patches，每个 patch/channel 用 12 个统计与时间特征。模型 hidden 32；跨 patch 传递 memory，patch 内按 T/S/ST 的关系处理并通信，汇总全部非空 patches 做分类。不是只用最后一个 state；patch 统计也不保留完整事件顺序。

本轮 10 个配置：

| 研究问题 | 配置 | 共用参照 |
|---|---|---|
| 历史 | history_both、history_reread、history_state | 比较两条历史通路 |
| 初始写入 | write_static、write_content | history_state |
| 交互 | interaction_no_communication、interaction_dynamic、interaction_preplanned、interaction_centralized、interaction_no_commit | history_state 与 dynamic |

`history_both` = carry state + 回读旧输入；`history_reread` = reset state + 回读旧输入；`history_state` = carry state + 不回读旧输入。state-only 的 ST 读取本 patch 初始写入之前冻结的其他 channels 旧 memory；不能让当前写入污染它。

`write_static` 用每角色固定 learned sigmoid；`write_content` 用当前 token、旧 state、role identity 控制初始 GRU 写入。独立 gate 允许同时写多个角色。关闭初始 gate 仍可能经后续 attention/消息读当前输入；不是信息完全隔离，也不减少实际计算。content 比 static 参数更多。

交互对照固定 history_state：固定 T→S→ST；无消息；反馈动态选下一角色/COMMIT；patch-start 预规划；中央 recurrent controller；动态但不许 COMMIT。中央 controller 增加 state/参数，不能声称完全容量匹配；逻辑 update count 不等于 FLOPs。

state-only 的完整计算仍包括 seen mask、clock、pooled readout。当前实现仍编码整段并保存 patch 输出，不宣称常数 RAM 或 Markov 最优。递归式不保证 learned state 是充分统计量。

本轮不包含 content gate + dynamic routing 同时开启的组合：先隔离各机制，之后依据 validation 决定是否组合。原始 12 variants、旧 pilot 分数保留用于审计，不进入新 `ablation` group。详细易读说明在 [method-ablations.md](method-ablations.md)。

## 理论和文献脉络

- [RIMs](https://arxiv.org/abs/1909.10893)：独立 recurrent mechanisms、输入相关激活、未激活模块保留 memory。
- [EntNet](https://arxiv.org/abs/1612.03969)：state/key 相关独立写入 gate，同一输入可更新多个 memory blocks。
- [Skip RNN](https://arxiv.org/abs/1708.06834)：选择更新或复制 state，研究更新预算；原 gate 不能描述成先读当前输入再决定不读。
- [Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf)、[CRU](https://proceedings.mlr.press/v162/schirmer22a.html)：在明确 latent/observation 假设下递归 filtering；不是任意 GRU 足够的证明。
- [Bahdanau attention](https://arxiv.org/abs/1409.0473)：固定向量压缩可能形成瓶颈，回读历史也有理由。

这些支持提出实验假设，不构成我们的优越性或新颖性证明。旧 recurrence-off 仍能读 input prefix；旧分数不下降只能说明允许回读时 carry state 没显示额外收益。

## 分阶段计划和协议

1. 本地测试和 10 配置 × 3 个真实 datasets 的一轮 smoke；CPU 验证，不冒充 H200 验证。
2. Orion：JapaneseVowels、CharacterTrajectories、Physionet2012 × 10 配置 × seed 0，30 次完整 validation pilot。
3. 所有 pilot 成功、数值/预测重建/动作覆盖/checkpoint/诊断/来源一致后，自动进入 18 datasets × 10 配置 × 3 seeds，540 次 official test fits。

最多 570 次新训练；正式阶段重新训练包含 pilot seed0。没有以分数高低挑配置，没有用 test 做门槛。epochs100、patience15、batch32、AdamW lr.001、weight_decay.0001、hidden32。validation macro-F1 选 checkpoint，显式 test 一次。每次记录源码、环境、配置、data/split 哈希。

数据 revision `f416b1b432f248a73323222f2378d0e8433ca6d2`；保留 irregular missing mask，train-only scaler/patch width/feature标准化；最后 patch 开放容纳超出 train horizon 的事件。label 编码只拟合 train。JapaneseVowels 时间沿用上游坐标，不冒充真实秒。

结果包括 gate mean/std/count、角色动作路径、COMMIT 比例、参数量和 runtime。`imts_agents compare` 只读 official test，缺失 seeds 明确 partial，不把失败记零分、不挑最好 seed、不混 validation/test。方法 expected540；baseline expected252。

## 提交器和额度

沿用并修改现有 `scripts/submit_method_pilot.py`，没有新建 nightly launcher。`--group ablation --staged --evaluate-test --submit --watch --baseline-root ~/IMTS --output-dir results/method-ablation-full` 驱动上述两阶段。

每 task 1 GPU、4 CPU、32GiB、3h；rolling refill 每30秒，按 global running+pending 总60补位，不等整波。当前 baseline旧watch仅跑GPU2；方法保护其下一波两名额，watch退出后释放。识别依赖同login node的 `/proc`；跨节点需显式 `--reserve-other-jobs 2`。

队列查询/回执不确定、源码或环境变化时停止进一步提交，避免重复或混版本。失败记录需检查后 `--retry-failed`，每 invocation 仅重试一次。sbatch jobs 在SSH断线/电脑关机后继续；nohup watcher 通常也继续，但login节点重启会影响未提交的后续任务。

完整部署和检查命令：[method-orion.md](method-orion.md)。

## Baseline 最新实际状态

以下是用户 **2026-10-14 21:10 SGT** 输出，之后没有新的服务器快照：

- **169/252 successful fits (67.1%)**；76/108 dataset/model groups complete。
- CPU四方法125/144；GPU两方法44/108。83 fits尚无成功，不全等于未提交。
- ROCKET48/54、BORF16/18、LGBM16/18、RIFC45/54、TIMESNET33/54、BRITS11/54。
- 28 failed attempts、25 unfinished JSON records；25不等于25个正在跑的jobs。
- 六方法全seeds齐：Mimic3、Animals、AsphaltObstaclesCoordinates。
- Pamap2、GeolifeSupervised全部模型尚无成功；原因尚未查日志，不能猜。
- 旧watcher PID26722继续；GPU162149_0/1运行19:36。不要把更早Wave37的32deferred当最新。

服务器结果 `~/IMTS/results/jv-first-full`，名称含JV但覆盖全部18。baseline seeds：ROCKET/RIFC/TIMESNET/BRITS为0,1,2，BORF/LGBM为0。252 fits不等于108groups；只有全部seed成功group才complete。

已知BRITS/Physionet2012曾因1h时限在epoch56终止。BORF/Physionet2012曾成功F1.508626、fit932秒。现有baseline继续旧时限/并发；本轮不更改它。

## 历史记录与文件定位

- [project-handoff-before-ablations.md](project-handoff-before-ablations.md)：本轮之前的完整讨论、文献、旧proposal、12variants分数、旧baseline接续备用工具。其中7配置/每批4/替换baseline watcher等已被本轮决定取代。
- [method-pilot-results.md](method-pilot-results.md)：旧12variants JapaneseVowels seed0 validation-only；不能据此宣称胜过baseline。旧dynamic/preplanned路径全部S→T→T、未用COMMIT，机制效果未成立。
- `src/imts_agents/model.py`/`variants.py`：关系、gate、交互和10配置。
- `src/imts_agents/data.py`：train-only事件预处理及内存保护。
- `src/imts_agents/runner.py`：训练、诊断和明确test边界。
- `src/imts_agents/report.py`：18dataset方法/六baseline正式结果汇总。
- `scripts/submit_method_pilot.py`：现有提交器的rolling/staged升级。

新chat开头可说：“请先读project-handoff.md，再核对代码commit和我最新的Orion输出。当前只研究历史、写入、交互，T/S/ST固定；方法最终全18，不替换旧baseline watcher。把本地验证、push和server运行分开，每次只解释一个机制。”

## 本轮验证与交付状态

代码检查与真实数据检查记录见 [method-validation.md](method-validation.md)。本轮 commit hash 见交付包 README 和 `git log -1`。本地提交不等于push；当前环境缺少GitHub写入凭据，使用git bundle交付分支，由用户push或直接导入Orion。没有实际向Orion提交新方法任务。
