# Method pilot：第一轮 2×2 实验结果

四个预先定义的 core 实验及八个 motivation controls 均已完成，共 12 次独立训练。当前结果是 **JapaneseVowels、seed 0、validation-only 的探索性 pilot**；validation 同时用于选择 checkpoint，official test 未用于训练、early stopping、policy reward 或模型评估。不能把这里的 validation 分数与 baseline 的 test 分数直接排名，也不能据此宣称超过 baseline。

## 本轮比较的两个改动

- **A：外层 recurrent by time patch**。开启时跨 patch 保留角色的 channel memory；关闭时每个 patch 重置这些 memory。关闭 A 的模型仍能读取当前已开放的 causal input prefix，因此这个控制隔离的是“递归状态传递”，不代表完全删除历史信息。
- **B：内层 T/S/ST**。开启时使用 T、S、ST 三个角色，固定顺序 T→S→ST，每个角色一次更新；关闭时使用一个 ALL-relation 角色，每个非空 patch 更新三次。四组均采用相同输入、head、训练规则和最大逻辑更新次数。动态行动策略属于后续 motivation 消融，没有混入这个 2×2。

## 实测结果

| Variant | A 外层 recurrence | B 内层 T/S/ST | Validation macro-F1 | Accuracy | 参数量 | Best / stop epoch | Fit 秒 | Validation prediction 秒 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `core_none` | 关闭 | 关闭 | 0.919960 | 0.926471 | 21,257 | 69 / 84 | 44.878 | 0.0599 |
| `core_outer` | 开启 | 关闭 | 0.904254 | 0.911765 | 21,257 | 49 / 64 | 35.862 | 0.0554 |
| `core_inner` | 关闭 | 开启 | 0.937722 | 0.941176 | 52,873 | 22 / 37 | 24.978 | 0.0800 |
| `core_both` | 开启 | 开启 | 0.937722 | 0.941176 | 52,873 | 20 / 35 | 24.669 | 0.0711 |

四组全部正常 early stopping。每个 validation 样本平均执行 **17.25 次逻辑 role update**，因为样本实际含有的非空 patch 数不完全相同。这个计数不包含 encoder、message、policy 的额外计算，不能等同于 FLOPs。

运行位置是这里的本地 CPU 环境，每个进程一个 torch thread；core 起初最多两个训练进程并行，末段与 motivation 实验重叠，本地最多四个训练进程。上述时间是实际运行记录，受共享 CPU 负载和 early stopping epoch 数影响，不是单独隔离后的效率 benchmark，也不是 Orion/H200 的耗时。

## 目前能说明什么

- 外层 recurrence 在本轮没有显示额外增益：single-ALL 控制中 `core_outer` 低于 `core_none`；T/S/ST 控制中 `core_both` 与 `core_inner` 的最佳 validation macro-F1 相同。
- 内层 T/S/ST 的初步表现较好，但参数量从 21,257 增至 52,873；下方补充了 `single_matched`、`homogeneous_roles` 控制。它们在本次 validation 上低于 T/S/ST，但单个 seed 与小 validation split 仍不足以证明因果解释或普遍增益。
- 只有一个 seed、68 个 validation 样本，而且 checkpoint 用这个 validation 选择。差异目前只用于确定下一步检查，不构成 statistically significant 的结论，也不足以证明 why agent、why multi-agent 或普遍有效的 irregularity 建模。

## 固定实验协议

| 项目 | 本轮设定 |
|---|---|
| 数据集 / seed | JapaneseVowels / 0 |
| 划分 | official 非 test pool 270 个样本，按同一 seed 作 25% inner validation：train 202、validation 68；official test 370 保留 |
| 输入 | 8 个 chronological patches，12 个 channels，每个 patch/channel 12 个统计与时间特征 |
| 预处理 | per-channel value scaling、patch width、feature standardization 全部仅拟合 inner train |
| 模型 | hidden 32；每个非空 patch 最大 3 次 relation update |
| 优化 | AdamW，learning rate 0.001，weight decay 0.0001，gradient clipping 5 |
| 训练 | batch size 32，最多 100 epochs，validation macro-F1 最大化，patience 15；相同分数保留更早 checkpoint |
| 评估 | deterministic eval；保存最佳 checkpoint 的 validation probabilities，重新读取并复算指标 |
| 环境 | PyTorch 2.6.0+cpu、NumPy 1.26.4、scikit-learn 1.5.2 |
| Test | 未执行；将来锁定配置后显式评估，不能根据 test 反复修改模型 |

数据方面也有明确限制。这里把 patch 内事件压缩为统计特征，不能恢复全部事件顺序。最后一个 patch 开放到无穷远，超出 training horizon 的事件会保留在最后一格而不是截断：本次 validation 有 1 个样本、12 个记录超出名义 horizon。JapaneseVowels 的原始数值型时间坐标没有物理单位属性，代码保留 upstream 的 1e-9 缩放，不据此宣称单位为真实秒。该数据集适合先检查可运行性，不能覆盖全部 uneven/asynchronous/missingness 动机。

## 审计与复查

- Source SHA256：`77c57ba1c8253555375bb96e5bd56da029386f42ede1153381a95f6c2a9c7313`
- Comparison protocol SHA256：`2783a8c3f08c57cd0426b843a22dd503c633b55fc89030c4707fab3d9e21d6c6`
- 数据 revision：`f416b1b432f248a73323222f2378d0e8433ca6d2`
- 数据文件 SHA256：`20f1909acb5bba1b680f7ef32f1ea3990ef5d7b9dc54fd1357a23e09c8e3ed06`
- 精确配置、split ID hashes、label hashes、prepared data hashes、checkpoint 与 prediction hashes 见 `references/method-pilot-summary.json` 和各 run 的 `result.json`。汇总中的数值直接读取原始 JSON，没有手工改写。

本次四个运行目录：

- `core_none`：`results/local-method-pilot/validation/JapaneseVowels/core_none/seed-0/20261013T141752Z-e7fc4498`
- `core_outer`：`results/local-method-pilot/validation/JapaneseVowels/core_outer/seed-0/20261013T141752Z-fed368f9`
- `core_inner`：`results/local-method-pilot/validation/JapaneseVowels/core_inner/seed-0/20261013T141850Z-86391e7f`
- `core_both`：`results/local-method-pilot/validation/JapaneseVowels/core_both/seed-0/20261013T141905Z-11197d31`

从仓库根目录查看所有当前结果：

```bash
.venv/bin/python -m imts_agents summary --output-dir results/local-method-pilot
```

该命令按 source、软件版本、数据与预处理、split、seed、训练配置和评估阶段区分 protocol groups。只在相同 group 内比较；`failed` / `running` 单独保留，重复运行逐项列出，不自动选最好的 test 或 seed，也不把失败记成 0 分。

复现某个已定义的 core 实验时，把 `core_both` 替换为对应 variant：

```bash
.venv/bin/python -m imts_agents run \
  --variant core_both --dataset JapaneseVowels --seed 0 \
  --device cpu --epochs 100 --patience 15 --batch-size 32 \
  --hidden 32 --n-patches 8 --max-inner-steps 3 --threads 1 \
  --cache-dir /workspace/scratch/dd6a79c01e9a/datasets \
  --output-dir results/local-method-pilot
```

每次调用会创建新的独立 run，不会续训或覆盖这里的 checkpoint。缓存路径需要按实际机器调整；它不是影响数据划分的超参数。

## Motivation controls：全部完成

以下八项与 core 表使用同一个 protocol、同一个 seed 和相同的 202/68/370 划分。这里的 best checkpoint 仍由 validation 选择，official test 未执行。

| Variant | 主要参照 | Validation macro-F1 | Accuracy | 参数量（其中 policy） | Best / stop epoch | Fit 秒 | 平均 role updates/sample |
|---|---|---:|---:|---:|---:|---:|---:|
| `no_communication` | `core_both` | 0.937722 | 0.941176 | 52,873（0） | 20 / 35 | 22.097 | 17.250 |
| `dynamic_feedback` | `core_both` | 0.937722 | 0.941176 | 62,677（9,804） | 34 / 49 | 81.138 | 17.250 |
| `preplanned_path` | `dynamic_feedback` | 0.936850 | 0.941176 | 62,677（9,804） | 46 / 61 | 95.845 | 17.250 |
| `centralized_router` | `dynamic_feedback` | 0.937722 | 0.941176 | 84,013（31,140） | 35 / 50 | 80.931 | 17.250 |
| `no_commit` | `core_both` | 0.924020 | 0.926471 | 62,677（9,804） | 22 / 37 | 67.695 | 17.250 |
| `no_time` | `core_both` | 0.937722 | 0.941176 | 52,873（0） | 19 / 34 | 22.774 | 17.250 |
| `homogeneous_roles` | `core_both` | 0.913394 | 0.926471 | 52,873（0） | 24 / 39 | 27.901 | 17.250 |
| `single_matched` | `core_both` | 0.921849 | 0.926471 | 52,879（0） | 62 / 77 | 63.628 | 17.250 |

`single_matched` 为 52,879 个有效参数，接近 `core_both` 的 52,873 个；这个控制匹配的是容量，不保证 FLOPs 相等。`homogeneous_roles` 与 `core_both` 参数量相同，把三个角色的直接读取范围都换成 ALL。`no_commit` 与固定 T→S→ST 的 `core_both` 都用满相同的 relation-update 次数，因此它们是动态选择与固定路径的更新次数对照；policy 额外开销仍需计入。

当前 observations：

- **通信和时间特征的必要性尚未得到支持。** `no_communication`、`no_time` 与 `core_both` 的最佳 validation macro-F1 相同。这只表示本次分数持平，不表示预测逐项相同，也不说明这些模块在其他数据集无用。
- **动态策略与多角色决策没有显出额外分数优势。** `dynamic_feedback`、`centralized_router` 都与固定 `core_both` 持平；`no_commit` 低于固定路径。distributed 与 centralized 的 policy 参数量也不同，不能把这个比较当作严格相同容量的证明。
- **动态模型没有在这个 checkpoint 上省下 role updates。** `dynamic_feedback` 最佳 checkpoint 在 68 个 validation 样本的平均 T/S/ST/COMMIT 次数是 11.5 / 5.75 / 0 / 0，总计 17.25 次，与固定三步相同。训练时曾探索的行动，不等于最终 deterministic evaluation 时真正执行的行动。
- **`preplanned_path` 与 feedback 的分数非常接近。** 单次最佳 validation 分数之间的微小差异不足以支持“必须根据中间结果逐步决策”。

八个 motivation run 的完整路径与校验哈希均已写入 `references/method-pilot-summary.json`。每组原始 run 保留 `history.jsonl`、`data_metadata.json`、`best.pt`、`val_predictions.npz`、`result.json`。这些结果用于决定下一阶段的诊断，不应直接推动 18 个数据集的大规模实验或宣布 why agent 动机已被证明。

## 保存 checkpoint 的行动路径诊断

这是在现有最佳 checkpoint 上进行的只读 validation trace，没有重新训练，也没有读取 test 分数。68 个 validation 样本共包含 391 个非空 patches：

| Variant | 确实执行的路径及 patch 数量 |
|---|---|
| `dynamic_feedback` | S→T→T：391 |
| `preplanned_path` | S→T→T：391 |
| `no_commit` | S→T→S：15；S→T→ST：376 |
| `centralized_router` | T→T→T：74；T→T→ST：51；T→ST→ST：79；S→ST→ST：187 |

**`dynamic_feedback` 和 `preplanned_path` 在全部 391 个非空 patches 上都执行 S→T→T。** 因而本轮它们的最终策略表现为固定路径，没有展示输入相关的行动变化；也没有使用 ST 或提前 COMMIT。不能把与固定模型持平的分数解释成 adaptive routing 或 feedback-driven refinement 已获验证。这个观察只针对本次数据、seed 和保存的 checkpoint，不表示模型结构没有学习动态行为的能力。

完整诊断：`results/local-method-pilot/diagnostics/routing-paths.json`；其中记录 checkpoint、原始 result、source、protocol 和 validation sample ID 的哈希。诊断文件校验值和计数也收录在 `references/method-pilot-summary.json`。
