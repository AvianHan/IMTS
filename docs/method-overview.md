> 历史说明：本文记录原始 12 个 variants。本轮已按用户决定保留 T/S/ST，使用新的 10 配置计划；请先读 [method-ablations.md](method-ablations.md) 和 [method-orion.md](method-orion.md)。

# 第一版方法与实验总览

这一版把两个主要改动分别实现为 **A：按时间坐标切 patch 并推进的跨 patch recurrent memory**，以及 **B：内层 T/S/ST 关系分工与通信**。先做四组 2×2 对照，再用八组 motivation 对照检查机制是否必要。

这是可训练的研究原型。代码实现了需要验证的机制；它们是否提高 classification、是否值得称为 agent 的增益，需要实验结果支持。本文不报告尚未取得的结果。

## 一次 forward 在做什么

1. 一个样本包含所有 channel 的带 timestamp 记录。用共同时间边界切成 `P₁ … Pₚ`；**一个 patch 包含该时间段的所有 channel**，不是一个 channel 一个 patch。
2. 将每个 patch/channel 的记录压缩成 12 个特征，再经过共享 token encoder 与 channel embedding。
3. 依次读取 patch。当前有记录的 channel 先通过 input GRU 将新输入写入各角色自己的状态。input GRU 的权重共享，角色持有的状态不同。
4. 内层执行关系读取、角色间消息交换和状态更新。四组核心实验均使用固定三次关系更新；动态行动是后续独立对照。
5. 每个 patch 的状态按有效 channel 和角色汇总，再汇总所有非空 patch，经过 classification head 得到 class logits。

开启 A 时，每个角色的 channel-wise state 会传给下一个 patch。关闭 A 时，在每个 patch 开始将 state 清零；**该角色仍可 attention 到当前及此前 patch 的输入 token**。这样比较的是“保留计算状态”的贡献，而不是同时剥夺历史输入。关闭 A 也不会取消 patch 内的三次状态更新。

T/S/ST 是整段样本中持续使用的三个模块。开启 A 后，它们的状态持续更新，不会为每个 patch 创建一套新模型。不同样本之间不共享 memory。

## T、S、ST 的具体含义

对于当前 patch `p`、channel `c` 的 query，直接读取范围如下：

| 角色 | 允许直接读取的 token | 直观含义 |
|---|---|---|
| T | 相同 channel，当前或此前 patch | 通道内的时间关系 |
| S | 不同 channel，当前 patch | 同一时间窗的跨通道关系 |
| ST | 不同 channel，严格早于当前 patch | 跨时间窗的跨通道关系 |
| ALL | 所有 channel，当前或此前 patch | 不做 T/S/ST 分工的对照 |

这三个范围覆盖 ALL，且不读取未来 patch。第一个 patch 的 ST 没有此前 token 可读。同窗中的不同 channel **不要求同时采样**。

这里 S 指变量／channel 之间的关系，不预设 channel 对应物理空间中的相邻位置。

每个关系模块有自己的 attention 与 GRU 参数；角色间利用 pooled state 的 query/key/value 交换消息。上述范围限制的是**直接 token 读取**，收到的消息可以携带其他角色的信息，因此不能把角色解释成完全隔离的信息源。

## 四组核心实验：分别加入 A、B

| 实验名 | A：跨 patch state | B：T/S/ST 分工 | 每个非空 patch 的内层计算 | 核心比较 |
|---|---|---|---|---|
| `core_none` | 重置 | 无；一个 ALL 模块 | 同一个 ALL 更新三次 | 共同起点 |
| `core_outer` | 保留 | 无；一个 ALL 模块 | 同一个 ALL 更新三次 | 对比 `core_none`：A 单独贡献 |
| `core_inner` | 重置 | 三个专门角色，允许通信 | 固定 T → S → ST | 对比 `core_none`：B 单独贡献 |
| `core_both` | 保留 | 三个专门角色，允许通信 | 固定 T → S → ST | A 与 B 的组合 |

这四组共享数据表示、split、hidden size、classification head 形式和训练规则。`core_none` 是本方法内部的共同起点，不能替代正在复现的 ROCKET、BORF、LGBM、RIFC、TimesNet、BRITS 六个外部 baseline。

开启 B 同时引入三个角色的参数容量，核心 2×2 的 B 效应不能直接归因于“分工本身”。后面的 `homogeneous_roles` 与 `single_matched` 分别检查角色分工与额外容量的解释。

除了比较“只加 A”“只加 B”，还要检查组合中的增量：`core_both − core_inner` 检查已有 B 时加入 A；`core_both − core_outer` 检查已有 A 时加入 B。组合分数最高并不自动证明二者存在正向协同；协同需要检查两者增量之差，并在更多 seeds、数据集上确认。

## 八组 motivation 对照

所有下列变体都开启跨 patch recurrence。除表中指出的改动外，沿用对应参考配置。

| 实验名 | 实际改变 | 主要参考实验 | 要验证的 motivation |
|---|---|---|---|
| `no_communication` | 固定 T → S → ST，角色间消息置零；最终 readout 仍汇总各角色 | `core_both` | 角色交换中间信息是否有贡献？ |
| `dynamic_feedback` | 每一步根据当前状态与消息选择 T/S/ST/COMMIT | `core_both` | 反馈驱动的动态行动在准确率与实际成本上是否有收益？此处只匹配最大更新次数 |
| `preplanned_path` | 所有行动分布都使用 patch 开始时冻结的 observation，可随步骤不同，但不读取中间执行结果 | `dynamic_feedback` | 决定下一步时，是否真的需要先前执行的反馈？ |
| `centralized_router` | 一个 recurrent controller 接收全部角色 observation，决定下一行动 | `dynamic_feedback` | 多角色分别产生决策意见，是否优于一个中央 controller？ |
| `no_commit` | 反馈驱动地选择角色，但禁止 COMMIT；每个非空 patch 必须执行三次更新 | `core_both`；另对比 `dynamic_feedback` | 前一对检查同更新次数下动态选择；后一对检查提前结束的作用 |
| `no_time` | 将标准化后的三个显式时间特征置零，保留其余特征与 mask | `core_both` | 显式时间位置与 time gap 是否有额外贡献？ |
| `homogeneous_roles` | 三个独立参数的 ALL 模块，保留通信和三次更新 | `core_both` | 收益来自 T/S/ST 分工，还是仅来自三个模块？ |
| `single_matched` | 一个 ALL 模块，加入实际参与 forward 的容量，近似匹配 `core_both` 参数量 | `core_both` | 收益是否只是参数更多？ |

动态版本中，三个局部 policy 各自产生 action logits，再以平均 logits 选择一个下一行动。training 使用 sampling 和 REINFORCE；validation/test 使用 argmax。选中的关系模块才执行该样本的关系更新。COMMIT 只能在至少一次关系更新后选择，表示结束当前 patch 的内层计算；默认最多三次行动决策。

这是一种明确的局部 policy 聚合实现，还不能据此声称已经证明 decentralized multi-agent 必要。特别是中央 controller 拥有全部角色 observation，但其 policy 参数量不同，比较时必须同时检查容量与耗时。

如果固定交互已经足够，就缩小“动态决策必要”的主张；如果预先决定路径同样有效，就缩小“中间反馈必要”的主张；如果中央 controller 同样有效，就不能仅凭三个角色声称多角色决策更好；如果同构角色或参数匹配单模块同样有效，就需要重新解释关系分工的价值。单个数据集、单个 seed 的差异只能用于下一步诊断。

## 数据表示：保留什么，简化什么

每个 patch/channel 的 12 个输入特征如下。只对有限观测值计算 value statistics；显式 NaN 记录另外保留存在信息。

| 类型 | 特征 | 数量 |
|---|---|---:|
| 观测数量 | `log1p_count` | 1 |
| Value statistics | `mean`、`std`、`minimum`、`maximum`、`first`、`last` | 6 |
| 显式时间 | 首条记录在窗内的位置、末条记录在窗内的位置、距此前最后一次有效观测的 `log1p` gap；都按 patch width 缩放 | 3 |
| 记录与缺失 | `log1p_record_count`、`observed_fraction` | 2 |

还有独立的 token mask：有存储记录，包括显式 NaN，mask 为 true；该 patch/channel 没有记录时为 false，并将特征置零。显式 NaN 不会重置“最后一次有效观测”的时钟。原始文件若没有记录某次缺失的 timestamp，代码不会虚构记录或推断缺失的物理原因。

所有 channel 使用一个样本共同的时间原点，保留相对起始时间差。默认八个 patch；patch width 由 inner training 的最大样本时长除以八确定。最后一个 patch 向后开放，保留较长 validation/test 样本超出的记录，并记录 overflow 数量。原始 value 的每通道标准化、patch 特征标准化、patch width 均只在 inner training 上拟合。

Pyrregular 适配层沿用源坐标乘 `1e-9` 的缩放。数值缩放本身不证明单位是秒；JapaneseVowels 等数据集的时间坐标可能是顺序索引，应读取并保留源坐标类型与单位记录，不能将它们直接解释成实际物理秒数。

`no_time` 仅移除表中的三个显式时间特征。它仍保留时间切窗、patch 顺序、观测数量、记录存在与缺失信息，不能称为“移除了所有时间信息”。

这一版将窗内事件压缩为 statistics，不能保留窗内完整事件序列。它用于先验证外层 recurrence 和内层关系机制，不是 TimeFilter、TarMAC 的完整复现，也不能直接代表最充分的不规则事件编码。对 irregularity 的机制判断还需要实际包含相应不规则性的其他数据集。

## Training、validation、test 与成本

| 项目 | 当前 pilot 约定 |
|---|---|
| 数据划分 | 保留官方 test；官方非 test pool 按 baseline neural wrapper 的规则再划分 25% validation，同 seed 使用相同 sample IDs |
| 默认训练配置 | hidden 32、8 patches、batch size 32、AdamW、learning rate 0.001、weight decay 0.0001、最多 100 epochs、patience 15 |
| Checkpoint 选择 | validation macro-F1 最高的最早 epoch |
| Policy 学习 | training cross-entropy 与关系更新成本形成 reward；不使用 validation/test 标签作为 policy reward |
| 默认运行 | 仅输出 validation 结果，标记为探索性 pilot |
| Test | 配置确定后，对锁定的 validation checkpoint 单独评估一次；校验代码、配置、数据与 checkpoint，重复或失败后自动重试会被拒绝 |
| 科学记录 | source/data/split/config/software fingerprints、checkpoint、predictions、逐 epoch history、指标与运行成本 |

pilot 的默认训练预算是本原型的起始设置，不等同于六个 baseline 的论文配置。选择配置看 validation；正式对比必须使用相同官方 test 和明确报告的训练协议。validation 分数不能与 baseline test 分数放在同一排行榜。

需同时查看 macro-F1、accuracy、参数量、policy 参数量、training 时间、inference 时间、显存、实际角色更新与 COMMIT 次数。失败或超时不能记成零分，smoke 分数不能当正式成绩，不能从重复尝试中按最高 test 分数挑选结果。

预算解释有四个边界：

- `core_both` 与 `no_commit` 匹配每个非空 patch 的三次关系更新；带 COMMIT 的动态版本只匹配更新上限。
- 一次关系更新不代表相同 FLOPs；不同关系范围、policy、通信和 input GRU 均有成本。实际延迟单独测量。
- `single_matched` 近似匹配参数量，不保证匹配 FLOPs 或延迟；核心四组中三个角色与一个角色的参数量也不同。
- `no_communication` 保留了未执行的通信模块权重，因此存储的 trainable 参数总数包括这些未使用权重，不能据此推断有效计算容量相同。

## 按阶段扩大实验

| 阶段 | 运行范围 | 扩大前检查 |
|---|---|---|
| 流程检查 | synthetic 或真实数据 smoke | shape/mask、forward/backward、有限 loss、输出记录、失败处理；不解释其分数 |
| 第一轮 pilot | JapaneseVowels、seed 0，先四组核心实验，再分批运行八组 motivation 对照 | 完整 training/validation 成功；查看 checkpoint、loss、行动分布、耗时与显存；修复机制错误后才扩大 |
| 重复性检查 | JapaneseVowels、seeds 0/1/2，固定待比较配置 | 不依赖单次 seed；重复训练样本仍使用同一协议 |
| 代表性数据检查 | 根据 irregularity、channel 数、样本量与序列长度选择少量数据集 | 验证不同数据形态及资源需求，不能只根据结果好坏挑数据集 |
| 完整 benchmark | 通过前述检查后再扩到 18 个数据集 | 固定配置与评估协议，合理 walltime，跟踪所有 seeds 和失败项 |

阶段之间需要检查结果，不会因为拥有更多提交名额就自动把所有数据集和变体全部启动。高额度用于并行运行当前已通过检查、已确定需要做的比较。

## 代码对应位置

| 文件 | 负责的部分 |
|---|---|
| `src/imts_agents/data.py` | 数据记录、缺失标记、时间切窗、12 维特征、训练集拟合的 preprocessing |
| `src/imts_agents/model.py` | 外层 state、T/S/ST scopes、通信、动态 action、readout |
| `src/imts_agents/variants.py` | 四组核心与八组 motivation 变体的唯一注册表 |
| `src/imts_agents/runner.py` | training、validation checkpoint、一次性 test 评估、结果与成本记录 |
| `scripts/run_method_pilot.py` | 本地按组依次运行，用于诊断与流程检查 |
| `scripts/submit_method_pilot.py` | Orion 小批量提交、额度检查、manifest 与提交 receipts |

这是独立 branch/worktree 中新增的方法实验族，结果也单独保存；已有 baseline 的训练实现与结果不应混为同一实验记录。
