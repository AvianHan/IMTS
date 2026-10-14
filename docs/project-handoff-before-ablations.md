# IMTS 研究与实验接续记录

更新：2026-10-14 21:13 SGT。最新服务器状态来自用户 21:10 SGT 的终端汇总；本地实现、讨论中的实验和服务器实际部署分开记录。

2026-10-14 21:03 SGT 接续方式更正：用户要求优先用现有 terminal 命令确认进度并调整现有脚本参数，不再把额外 launcher 当作必需部署步骤。此前已确认 CPU+GPU 合计 60 个提交名额，无需无故重复查额度；现有 baseline 脚本支持 `--cpu-batch-size`、`--gpu-batch-size` 与对应 concurrency。先取得成功覆盖率、当前 queue 和剩余 CPU/GPU 数量，再给出直接接续命令。后文独立接续工具是已准备的备用方案，不代表用户已接受部署或服务器已启用。

2026-10-14 21:10 SGT 用户实际汇总：169/252 successful fits（67.1%），76/108 dataset/model combinations complete；28 failed attempts、25 unfinished records。83 个 fits 尚无成功，不能全算未提交。CPU 四方法共 125/144 成功，GPU 两方法共 44/108 成功；分模型为 ROCKET 48/54、BORF 16/18、LGBM 16/18、RIFC 45/54、TIMESNET 33/54、BRITS 11/54。六方法全 seeds 已齐的数据集是 Mimic3、Animals、AsphaltObstaclesCoordinates；Pamap2 和 GeolifeSupervised 的全部模型均尚无成功记录，失败原因待查。队列实际仅 162149_0、162149_1 两个 GPU task，R 19:36；watcher 26722 仍运行。25 unfinished 不等于 25 个运行任务。最新日志尾部未给 deferred 数，不沿用旧 32 作为最新值。

2026-10-14 21:13 SGT 用户要求：先用易懂方式整理目前方法，共同讨论并确定实验，再在独立 branch 今晚启动。本地已有 feature/our-method；未确认它已 push 或部署 Orion。当前仅核对代码、结果和提出计划，没有启动方法实验。候选第一阶段是 7 个配置：单 ALL 的 state carry × 历史回读 2×2 共 4 组，加 T/S/ST 的 reset+prefix、carry+prefix、carry+state-only 共 3 组；先在多个数据集 seed 0 验证。该计划尚未获用户确认。角色选择性初始写入建议独立下一阶段，在固定历史方式下加静态 gate 与内容相关 gate 对照。现有 submit_method_pilot 每批硬上限 4，不能声称它已经能自动提交完整多数据集夜间计划。

## 先读这一页：我们现在在哪里

我们研究不规则多变量时间序列分类。当前原型有两个主要设计：按时间 patch 递归传递 memory；在每个 patch 内用 T、S、ST 三种关系处理信息。先验证机制是否有用，再扩大实验。

| 事项 | 真实状态 | 接下来 |
|---|---|---|
| 六个 baseline、18 个 datasets | Orion 仍在跑；旧 launcher 每波 GPU 2、CPU 4，未部署此前的额度升级 | 使用独立接续工具提高到 CPU+GPU 合计 60 个提交名额，检查成功覆盖率与失败原因 |
| 我们的方法及 12 个 variants | 已实现，完成本地 JapaneseVowels、seed 0、validation pilot | 先诊断关键机制，再扩展多个 datasets；最终与六个 baseline 在相同正式协议下比较 |
| state-only 历史传递 | 用户同意作为认真考虑的主架构候选；尚未实现/训练 | 拆开递归 state 与直接回读历史两条通路 |
| 各角色选择性接收新输入 | 已查代码与文献；尚未实现/训练 | 先比较广播初始写入与角色选择初始写入，其他机制固定 |
| 新 chat 接续 | 本文是项目记录，另提供可打开的持久页面 | 新 chat 先读取记录，再核对最新代码与终端状态 |

JapaneseVowels 只是小规模调试起点。不能把一个数据集的结果当作最终研究结论。不能把方法的 validation 分数与 baseline 的 test 分数直接排名。

## 沟通与协作约定

- 用中文解释，保留 training、validation、test、forward、head、memory、patch 等必要英文术语。
- 用户明确反馈：此前 overview 太密、难理解也看不下去。以后一次讲一个机制，先给具体例子，再说明它在流程中的位置，最后才给公式或完整实验表。
- 每次区分：讨论中的想法、已实现、已本地验证、已部署 Orion。不能把准备好的文件说成服务器已经启用。
- 先小规模验证，再分阶段扩大；不一开始把所有新设计全跑，避免错误浪费大量计算。
- 使用已确认的最大任务额度。没有方法任务运行时，不为方法空留名额。提交名额与实际获得 GPU 数量不同。
- 交付代码时给出 commit message，并明确是否已经 commit、push、部署。此前用户因没有 commit message 而漏掉 push。
- 在同一 GitHub repo 的不同 branch/worktree 开发方法，保留正在跑的 baseline source、configs、环境与结果。
- 重要讨论、实现、失败、协议和运行状态更新进本文/对应持久页面。新 chat 不能假设自动拥有整段旧聊天，用户应提供页面链接或本文件。

## 方法：先理解输入如何走

一个 patch 是一段时间内的观测。当前代码先把每个 channel 在这一段内的观测编码成 token，再更新三种角色的 memory，然后执行关系计算与通信，最终汇总做分类。

| 角色 | 直接读取的关系 | 直观理解 |
|---|---|---|
| T | 同一 channel、当前及过去 patch 的 token | 这个变量随时间怎样变化 |
| S | 当前 patch 的其他 channels | 同时出现的不同变量怎样关联 |
| ST | 严格过去 patch 的其他 channels | 别的变量较早的变化，与现在有什么联系 |

这些是直接读取范围，不是信息完全隔离：角色之间还可以通信，memory 也可能已经融合其他信息。

### 当前三个角色是不是都读新输入？

是。`src/imts_agents/model.py` 中，当前有效 channel token 经 `expand` 送给全部角色，并使用共享的 input GRU 更新各自 memory。第一个 patch 的旧 memory 均为零，因此初始写入结果相同；后续旧 memory 不同，GRU 更新可以不同。三个角色后面的关系计算不同，所以不能说最终表示始终相同。

当前 dynamic policy 选择的是后续执行哪个 relation update，不控制这次最初的输入写入。GRU 本身已经有 gate；新设计不能被描述成“原模型完全不能忽略无关输入”。

### 用户新想法：这条信息要不要写入这个角色？

例如新出现一段温度变化，T 可能需要更新趋势 memory；S、ST 是否也值得更新，交给学习到的 gate 判断。这只是解释例子，不预先规定温度一定只对 T 有用。

建议先做角色相关的初始写入 gate。gate 看当前 token、该角色旧 memory 和角色身份；每个角色独立决定写入强弱。同一条信息可以对多个角色有用，所以先不强制三选一。gate 接近零时，这次初始写入尽量保留旧 memory，而不是清零。

初步研究顺序：先固定历史访问方式与 T→S→ST 顺序，只比较广播写入和选择写入；加一个不依赖输入内容的静态 gate 对照，判断收益是否真的来自内容选择。检查 gate 是否随样本变化、是否长期只偏爱同一角色。若有效，再研究 hard routing 和同预算随机/固定选择。

边界：仅 gate 初始写入，后续 attention 仍可能直接读到相同 token，通信也可能传来该信息。因此它是“选择性初始写入”，不能称“该角色完全看不到输入”。真正的输入隔离需要额外定义 attention、policy 和通信旁路。soft gate 通常仍计算全部候选更新，不等于省 FLOPs；只有实际跳过计算并测量 latency/显存，才可讨论效率收益。

## 文献依据与尚未证明的部分

| 文献 | 对我们有用的内容 | 不能直接推出什么 |
|---|---|---|
| [RIMs，ICLR 2021](https://arxiv.org/html/1909.10893v6)，§2.2–2.4 | 模块用自身 state 查询当前输入，竞争激活；未激活模块保持 memory，活跃模块可读其他模块记忆 | 模块化假设与实验支持不等于我们在 IMTS 上一定更好；也不自动证明需要 agent 叙事 |
| [EntNet，ICLR 2017](https://arxiv.org/html/1612.03969v3)，§2.2 | 旧 memory 与固定 key 共同控制独立 sigmoid 写入 gate；同一输入能更新多个 memory blocks | 原实验是文本/world-state tracking；迁移到 T/S/ST 是我们的研究建议 |
| [Skip RNN，ICLR 2018](https://arxiv.org/pdf/1708.06834)，§3 | 在更新 state 与复制旧 state 之间选择，并研究更新预算 | 原 gate 根据此前 state 决定跳过；不能说它先看当前未读取输入，再判断内容价值 |

研究假设是：与某角色无关的新输入可能干扰其 memory，选择写入可能帮助分工。效果必须用多数据集对照来验证；这些文献不是现成的优越性定理，也不是新颖性证明。

## 已同意：把两条历史通路分开

现有 A=外层 recurrence 开关只控制跨 patch 传递角色 state。即使 A 关闭，模型仍能读取此前 causal input prefix。因此此前关闭 recurrence 后分数没下降，只能说明“允许回读时，递归 state 没显示额外收益”，不能说明历史记忆没有用。

| 跨 patch 保留 state | 直接回读历史 token | 要回答的问题 |
|---|---|---|
| 是 | 是 | 现有两条历史通路并存 |
| 否 | 是 | 只靠重新读取历史是否够用 |
| 是 | 否 | 只经压缩 state 传递历史是否有效、是否更省开销 |
| 否 | 否 | patch 局部处理的边界对照 |

先用单 ALL 模块做干净的 2×2，再在 T/S/ST 上比较关键版本。不能简单删掉历史 token，否则 ST 会失去其历史输入。候选 state-only 版本让 ST 读取其他 channel 在进入当前 patch 前冻结的旧 state；不能读取已混入当前 patch 的 state。

保持已有 pooled readout 和训练规则不变，避免同时改为只用最后 state。pooled readout 可用 running sum/count 表示，它们、clock、seen mask、必要 router state 也属于完整计算 state。

state-only 的意思是此前信息通过记忆延续，不是只保留上一段原始输入。它更贴近外层 recurrent 动机，但压缩可能丢失历史信息。递归表达式不自动证明原始观测满足一阶 Markov 性，也不保证有限维 learned state 是分类的充分统计量。

理论参考：[Särkkä, Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf) 的 Bayesian filtering 在指定 latent Markov/observation 假设下递归更新后验；[CRU, ICML 2022](https://proceedings.mlr.press/v162/schirmer22a.html) 将连续时间 latent dynamics 与递归 filtering 用于不规则时间序列。它们支持研究这种结构，但不能证明任意 GRU state 足够。[Bahdanau et al.](https://arxiv.org/abs/1409.0473) 的 attention 工作提醒固定向量压缩可能形成瓶颈，历史回读也有合理依据。

## 实现清单与关键协议

代码 repo：`https://github.com/AvianHan/IMTS.git`。方法分支：`feature/our-method`。方法主体提交：`7558939`，commit message：`feat: add recurrent T/S/ST method, ablations, and quota-aware pilots`。这是本地提交；本记录没有证据证明已 push 或部署 Orion。

工作副本：`/workspace/scratch/dd6a79c01e9a/IMTS-method`。原 baseline 工作副本：`/workspace/scratch/dd6a79c01e9a/IMTS`，原有五个 bounded-watch 修改需保留。服务器 baseline：`~/IMTS`，用户此前拉取到 `b55c720`。

| 文件/目录 | 作用 |
|---|---|
| `src/imts_benchmark`、`configs` | 六个 baseline 与固定配置，方法开发未改训练实现 |
| `src/imts_agents/data.py` | irregular records → patches/features，train-only preprocessing |
| `src/imts_agents/model.py` | recurrent memory、关系读取、通信与 policy |
| `src/imts_agents/variants.py` | core 与 motivation variants |
| `src/imts_agents/runner.py` | training、validation、显式 test、结果与 provenance |
| `scripts/run_method_pilot.py`、`scripts/submit_method_pilot.py` | 本地和 Orion pilot |
| `scripts/submit_parallel_baselines.py` | 用独立 launcher 驱动原 baseline checkout |
| `scripts/resume_max_baselines.py` | 本次新增的安全交接与 60 任务额度工具 |
| `docs/method-overview.md`、`docs/method-pilot-results.md` | 原详细 overview 与完整本地结果 |
| `references/method-pilot-summary.json` | 原始运行定位、数值、校验哈希 |

当前原型用 8 个 chronological patches；每个 patch/channel 12 个统计与时间特征，hidden 32。patch 内统计压缩不能恢复完整事件顺序。原型并非完整复刻其他论文模型。

数据 revision 固定为 `f416b1b432f248a73323222f2378d0e8433ca6d2`。保留原 records 中的 NaN，缺失位置另有 mask。scaler、patch width、feature standardization 仅拟合 inner train。最后一个 patch 开放，超出 train horizon 的记录仍保留。JapaneseVowels 时间坐标沿用上游 1e-9 缩放，不据此宣称真实秒。

训练设定：seed 0，batch 32，最多 100 epochs、patience 15，AdamW lr 0.001、weight decay 0.0001，hidden 32。JapaneseVowels official 非 test 270，内部分 train 202/validation 68，official test 370 留出。用 validation macro-F1 选 checkpoint，test 不能反复指导设计。

core 四组：`core_none` Aoff/Boff；`core_outer` Aon/Boff；`core_inner` Aoff/Bon；`core_both` Aon/Bon。Aoff 仍可回读历史。Boff 是单 ALL 模块，每非空 patch 固定三次更新；Bon 是 T→S→ST 各一次。读出汇总全部非空 patches 的角色/channel state，并非只用最后 state。

已实现八个 motivation controls：`no_communication`、`dynamic_feedback`、`preplanned_path`、`centralized_router`、`no_commit`、`no_time`、`homogeneous_roles`、`single_matched`。dynamic policy 的动作是 T/S/ST/COMMIT，至少一次更新、上限三次；train REINFORCE，eval argmax。角色 policy logits 平均得到共识；centralized policy 参数更多，并非容量完全匹配。

`no_time` 关闭三项明确时间特征，仍保留 chronology。`homogeneous_roles` 是三个 ALL 角色，参数量与 T/S/ST 相同。`single_matched` 参数差小于 1%。逻辑 role-update 次数不包括 input GRU、policy、message，不能等同 FLOPs。

## 已完成的本地 pilot：不要重复误解

12 次训练全部成功；地点是本地 CPU，不是 Orion/H200。它们是单数据集、单 seed、validation-only，尚未执行 official test。原实现阶段 174 项 tests passed；本次新增 20 项接续工具测试，完整测试套件为 194 passed，另经独立代码审查。Slurm 与进程交接分支通过 mock 测试；没有在本地冒充实际 Orion 提交验证。

| Variant | Validation macro-F1 | 主要观察 |
|---|---:|---|
| core_none | 0.919960 | 单 ALL，无 state carry，仍能回读历史 |
| core_outer | 0.904254 | 本轮 state carry 未提高分数 |
| core_inner | 0.937722 | T/S/ST，无 state carry |
| core_both | 0.937722 | 与 core_inner 持平 |
| no_communication | 0.937722 | 本轮未显示通信收益 |
| dynamic_feedback | 0.937722 | 最终路径全部 S→T→T |
| preplanned_path | 0.936850 | 最终路径也全部 S→T→T |
| centralized_router | 0.937722 | 四种路径，未提高分数 |
| no_commit | 0.924020 | 用满三步，不含提前结束 |
| no_time | 0.937722 | 本轮未显示这些时间特征收益 |
| homogeneous_roles | 0.913394 | 同参数量的三个 ALL |
| single_matched | 0.921849 | 与 core_both 近似同参数量的单模块 |

core_none/core_outer 参数量 21,257；core_inner/core_both 52,873。原四组 B 的差异混有容量变化，补充 matched controls 后也不能凭一次小 validation split 宣称普遍有效。

68 个 validation 样本共 391 个非空 patches。dynamic_feedback 和 preplanned_path 全部 S→T→T，没有使用 ST 或 COMMIT。它们没有展示输入相关路径变化，不能把持平解释成 adaptive routing 已证实。平均 17.25 次 role updates/sample，动态策略没有省下这个计数。

结果路径：`results/local-method-pilot/validation/JapaneseVowels/...`；诊断：`results/local-method-pilot/diagnostics/routing-paths.json`。Source SHA256：`77c57ba1c8253555375bb96e5bd56da029386f42ede1153381a95f6c2a9c7313`。Protocol SHA256：`2783a8c3f08c57cd0426b843a22dd503c633b55fc89030c4707fab3d9e21d6c6`。

## Baseline 的完整范围与运行沿革

正式范围是 18 datasets × 六模型，共 108 个 dataset/model combinations、252 次 fits。ROCKET、RIFC、TIMESNET、BRITS 用 seeds 0/1/2；BORF、LGBM 用 seed 0。

数据集顺序：Mimic3、Physionet2012、Physionet2019、CharacterTrajectories、Ldfpa、Pamap2、Animals、AsphaltObstaclesCoordinates、AsphaltPavementTypeCoordinates、AsphaltRegularityCoordinates、GeolifeSupervised、Seabirds、Taxi、Vehicles、InsectWingbeat、JapaneseVowels、Garment、SpokenArabicDigits。

结果在 `~/IMTS/results/jv-first-full`。名称含 JV，但里面运行全部 18 datasets，不仅 JapaneseVowels。

- 初始已有六次兼容成功，剩余 246 = CPU 140 + GPU 106。
- 2026-10-13，140-task CPU array 即使 `%4` 仍因 `QOSMaxSubmitJobPerUserLimit` 被拒；并发限制不等于提交任务数限制。回执在 `results/accounting/batches/20261013T093211Z-bc81dd88/cpu.receipt.json`。
- 更新到 `b55c720` 后改为每波 CPU 4/GPU 2；nohup watcher PID 26722，自 10-13 17:57:45 SGT 开始，日志 `results/accounting/batch-watch-20261013-175745.log`。
- BORF/Physionet2012/seed0：array task `160681_3`，success，macro-F1 0.508625945、fit 932.37 秒、Slurm elapsed 23:20。
- BRITS/Physionet2012/seed0：`160727_1`，到 epoch 56 被一小时 walltime 终止，不能算成功。
- 2026-10-14 最新用户快照：旧 watcher 仍为 PID 26722；Wave 37，GPU array `162140` 两个 task 运行，32 still deferred。日志已等待旧 array `162115` 很久后进入此波。

32 deferred 仅表示当前 watcher 尚未尝试的 tasks，不包括此前失败/超时。队列没有 CPU 任务也不代表全部 CPU 实验成功。watch finished、squeue 空、进程结束均不能代替成功覆盖率检查。

## 算力额度与本次接续工具

用户提供的 `sacctmgr` 显示 highpriority `MaxSubmitJobsPU=60`，已显示的其他 QoS/association limit 字段为空。60 是 CPU/GPU 合计 running+pending array tasks 上限，不是 60 块专属 GPU，也不能从空字段推断任何隐藏政策不存在。

用户提供的 h200p 配置：13 nodes、104 H200 总资源，MaxTime 3 days。它们是分区总量，不是用户独占。此前 `(Priority)` 说明排队优先级也会影响等待。

本次工具以三个独立 scripts 交付，放在 baseline repo 之外。继续用 `~/IMTS` 的 `.venv`、训练代码、configs、templates、结果和 accounting。

接续步骤：默认 dry-run；显式 `--submit --handoff` 时确认 QoS 60 和 CPU 2h/GPU 3h 的 sbatch test-only，验证旧 watcher 的 UID/cwd/argv/starttime，只停止该 watcher，不取消已提交 Slurm jobs；等待已登记 arrays 完整终态 accounting，再运行升级 launcher。

新配置：CPU/GPU 各池 batch 与 concurrency 上限 60，**combined total 60**、max-user-jobs 60；无方法任务时不预留四个名额。重用兼容成功；之前失败/超时可在新 watch 中再尝试一次，不是 checkpoint resume。GPU 3h 是增加时限，不是保证所有任务三小时内结束。

已知限制：仍按 wave 等整波结束，不是完成一个立即补一个的 rolling scheduler。其他用户自己的 jobs 也计入 60；若占用使整批越限，launcher 会停止，不能保证自动缩到剩余额度。Slurm 分配和队列优先级仍决定实际同时运行数。没有证据证明此工具已在 Orion 执行；最新实际状态仍是旧 GPU2 watcher。

有模糊提交回执、多个 watcher、不能确认 accounting、source/environment 变化时拒绝重复提交。不要删除 ledger/receipt 或直接重提来绕过检查。原来的明确 QoS 拒绝由已知 rejection 检测处理。

## 怎样看 baseline 真正完成了多少

在 Orion：

```bash
cd ~/IMTS
pgrep -af '[s]cripts/(submit_benchmarks|submit_parallel_baselines|resume_max_baselines)\.py.*--submit'
squeue -r -u "$USER"
.venv/bin/python -m imts_benchmark summarize \
  --results-dir results/jv-first-full \
  --output "results/accounting/benchmark-summary-$(date +%Y%m%d-%H%M%S).csv"
```

汇总输出 Coverage：目标 108/108 complete、252/252 successful fits。它会显示 macro-F1、缺失 seed、失败 attempts；失败不按零分平均。`running` JSON 可能是被 Slurm 强制终止后的遗留记录，要与 sacct 和日志一起判断。重复成功、混合不兼容代码/环境会报错，应先调查而不是自动挑最高分。

旧 watcher 日志：`tail -n 40 results/accounting/batch-watch-20261013-175745.log`。升级后的日志使用新名称，旧日志停止增长是交接后的正常表现。nohup 已脱离本地终端，关闭自己的电脑通常不影响服务器进程或 Slurm jobs；不能保证登录节点永不重启或作业一定完成。

## 接下来按这个顺序推进

1. 把已准备的额度升级真正部署 Orion，核对新日志与展开的队列；检查 baseline coverage 和失败清单。
2. 向用户用一个具体 patch 讲清输入、memory、T/S/ST 的位置；不要再直接塞一整张复杂 overview。
3. 先实施已同意的历史通路对照，保持其他机制固定；再独立测试角色选择性初始写入。当前二者尚未实现。
4. 在多个有不同不规则特征的数据集上做小规模 validation pilot，再扩大 seeds 与全部 18 datasets。不能只挑表现好的数据集。正式比较保持 split、preprocessing、metrics、训练/调参预算及 test 规则一致。
5. 单独检查 gate/path 是否真的随输入变化、ST/通信是否有贡献；无证据的 motivation 如实否定或缩小。
6. 锁定方法与评估协议后做正式 test，对照六个 baseline；同时测真实时间、显存和计算量，不仅数逻辑更新次数。

新 chat 可这样开头：“请先读 IMTS 研究与实验接续记录，再核对最新代码和我贴出的 Orion 状态。把已实现与待做分开；每次只解释一个机制。先继续未完成的额度接续/历史通路/输入写入任务，不要重跑已有 pilot 或据 validation 宣称赢过 baseline。”
