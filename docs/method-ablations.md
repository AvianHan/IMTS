# 本轮方法实验：一次看一个问题

本轮保留 T/S/ST 分工，不再比较“为什么要有三个角色”。原始 12 个 variants 和结果留作历史记录，今晚使用 `--group ablation` 的 10 个配置。

## 1. 历史怎样保留

先把它想成读下一页材料：可以把前面的原文重新翻出来，也可以只带着上一次整理的笔记。

| 配置 | 带旧 state | 回读旧输入 | 问题 |
|---|---|---|---|
| `history_both` | 是 | 是 | 两条历史通路同时存在 |
| `history_reread` | 否 | 是 | 重新读原文是否已经够用 |
| `history_state` | 是 | 否 | 只靠递归笔记能否保留有用历史 |

`history_state` 是后续两组共用的参照，不是已经被证明最好的主方法。它的 T 读当前同 channel 输入和旧同 channel memory；S 读当前其他 channels；ST 读进入当前 patch **之前冻结**的其他 channels memory。ST 不会把本 patch 刚写进去的信息当作旧历史。

角色间通信可以传递当前信息，因此直接读取范围不等于完全隔离。seen mask、时间信息、各 patch 的 pooled readout 也会保留；这是 state-only 的历史读取设计，不是原始序列满足 Markov 性的证明，也不是已经实现了常数内存的流式系统。

## 2. 新信息应该写给谁

在 `history_state` 的基础上，只改变最初写入 memory 的步骤，固定 T→S→ST 更新顺序。

| 配置 | 新输入如何写入 |
|---|---|
| `history_state` | 每个角色都进行原有 GRU 写入 |
| `write_static` | 每个角色学一个固定写入强度，不随输入变化 |
| `write_content` | 根据当前输入、该角色旧 state 和角色身份决定写入强度 |

这是独立 sigmoid gate：同一条信息可以同时写给多个角色，不强制三选一。gate 接近零时，初始写入保留旧 memory。

gate 只管初始写入；后续 attention 和消息仍可能读到当前信息。soft gate 也仍计算候选更新，不能宣称省计算。content gate 比 static gate 参数多，结果应连同参数量、gate 变化和预测表现一起判断。

## 3. 角色怎样交互

这组固定 state-only、原有输入写入。`history_state` 的固定顺序再次作为共用参照。

| 配置 | 改动 | 主要对照 |
|---|---|---|
| `interaction_no_communication` | 角色间消息置零 | `history_state`，看通信的作用 |
| `interaction_dynamic` | 根据当前中间结果选择下一角色，可 COMMIT | `history_state`；还要对照 no_commit |
| `interaction_preplanned` | 用 patch 刚开始的观察决定整段路径 | dynamic，看中间反馈的作用 |
| `interaction_centralized` | 一个 recurrent controller 决策 | dynamic，比较集中与分布决策 |
| `interaction_no_commit` | 动态选角色，但固定执行三步 | dynamic 看提前结束；history_state 看同更新次数的角色选择 |

COMMIT 表示提前结束当前 patch 的内层更新。centralized controller 多了自己的 recurrent state 和参数，并非严格相同容量。no_communication 保留原参数但不使用消息。逻辑 role-update 次数不等于 FLOPs。

本轮不再增加 ALL、角色数、同构角色、no_time、single_matched 等对照。旧配置仍可复查，但不进入本次提交清单。这里没有同时打开 content gate 与 dynamic routing 的组合实验；先分清两个机制各自的作用，之后再决定是否组合。

## 4. 今晚按什么顺序跑

1. 本地：代码测试与真实数据 smoke，检查形状、梯度、读写语义和记录文件。
2. Orion 检查阶段：10 配置 × JapaneseVowels、CharacterTrajectories、Physionet2012 × seed 0 = **30 次完整 validation 训练**。最多 100 epochs、patience 15；不是一轮 smoke。
3. 检查全通过：同一份冻结源码和环境，10 配置 × **18 datasets** × seeds 0/1/2 = **540 次正式 test 训练/评估**，与六个 baseline 比较。

检查阶段只以完成状态、有限数值、checkpoint、预测重建、动作记录和文件校验为门槛，不按哪个分数高筛选配置。正式阶段重新训练，包括检查阶段的 seed 0；总计最多 570 次新训练。test 在 validation 选定 checkpoint 后评估，不能拿 test 调参。

每个结果有 `result.json`、`best.pt`、validation/test 预测文件和 `diagnostics.json`。诊断包含各角色初始写入 gate 的 mean/std/count、动作路径和 COMMIT 使用率。分数持平但 gate/path 没变化时，不能宣称学会了内容选择或自适应交互。

## 5. 和 baseline 怎么比较

正式范围与 baseline 一致：18 datasets、相同官方 test IDs 和固定数据 revision。方法每配置 3 seeds；baseline 保持各自原协议：ROCKET、RIFC、TIMESNET、BRITS 三 seeds，BORF、LGBM 单 seed。

`python -m imts_agents compare` 只汇总成功的 official test。缺失 seed 明确标为 partial；失败不记零分，不挑最高 seed，不将 validation 分数混入正式排名。不同硬件上的运行时间仅作描述，不能直接宣称加速。

操作命令见 [method-orion.md](method-orion.md)。当前决策、baseline 实际进度和跨 chat 接续见 [project-handoff.md](project-handoff.md)。
