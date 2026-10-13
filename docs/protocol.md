# 实验协议与明确差异

## 数据

从固定 data-v1 revision 加载官方 HDF5；Pooch 校验注册 SHA-256。18 个数据集见 `configs/datasets.json`。保持默认 classification task、样本顺序和原 train/test；若文件包含 val/validation，沿用官方 `split != test` fitting 规则，并记录原始标签数量。

`irr.to_dense()` 产生 N×C×Lmax，sample 内 timestamps 转为位置，NaN 保留。六种方法都不追加真实时间 channel。dense allocation 前按 sample 内最长长度估算 X、T、split copies 的内存；超过默认 8 GiB 停止并说明原因，不悄悄截断。模型自身和稀疏原数据占用不包含在此预算。

## 方法

| 方法 | 官方设置 |
|---|---|
| ROCKET | MiniRocketMultivariateVariable(pad_value_short_series=0)，其余固定版本默认值；LGBM n_jobs=1 |
| BORF | aeon BORF 默认值 → float64 → LGBM n_jobs=1 |
| LGBM | Tabularizer → LGBM n_jobs=1 |
| RIFC | n_intervals=log；NaN-aware mean/std/min/max/median/skew/kurtosis；LGBM n_jobs=1 |
| TimesNet | n_layers=2, top_k=3, d_model=64, d_ffn=128, n_kernels=3, batch_size=32, epochs=1000, patience=50 |
| BRITS | rnn_hidden_size=256, batch_size=32, epochs=1000, patience=50 |

Random-state 参数、Python/NumPy/torch seed 显式设置。library 隐含默认值由依赖版本固定；每次 run 记录 estimator 暴露参数和所有 installed package versions。seed 不能保证不同 GPU、library 或硬件间 bitwise 一致。

BRITS/TimesNet 内部 validation 来自 sklearn 默认 25% split，先 stratify；上游遇到 ValueError 时会退回非 stratified split。原始 test 始终留出。两种 neural 方法无外部超参数搜索。

## BRITS compatibility fix

PyPOTS 1.5 `classification/brits/model.py:BRITS.fit` 在 `if val_set is not None` 分支中错误传入 `DatasetForBRITS(train_set, ...)`。这让 early stopping 的 validation loader 读取 training 样本。

`compat.BRITSHoldoutValidation` 复制官方 fit，唯一行为改动是上述参数改为 `val_set`；保存 BSD attribution。版本与原 fit token hash 必须匹配，否则明确报错而非自动套用补丁。测试捕获真实 DataLoader 的内容，验证 train/validation 没混用。结果记录 compatibility flag。

源码指纹使用 token 名称和文本（`token-names-v1`），忽略空白与注释，已核对 Python 3.10、3.11、3.12 一致。初版误用 token 数字编号，在 Python 3.10/3.11 上会因 `OP=54`（3.12 为 `55`）误报；修复没有关闭源码检查，也没有改变模型或 validation 修正。

这是为保证 held-out validation 正确而做的公开差异，不能声称历史作者一定采用或一定没采用同一 bug。测试分数也不证明已复现论文。

## 评估

主指标 macro-F1；另记 accuracy、macro precision/recall，使用 sklearn `zero_division=0` 和默认 true/pred 标签并集。概率列按 estimator.classes_ 解码，验证 shape、有限值及行和。逐样本预测落盘后重读并复算，防止只保存终端显示的数字。

fit_seconds 只包住 fit；predict_seconds 包住 predict_proba；初始化、下载和数据转换属于 total_seconds。这些计时定义可检查，但论文没有公开完整 runner，故不能保证历史计时边界完全相同。

汇总时仅纳入完整 benchmark 的成功记录；失败项不记 0；smoke 不进入排名。缺少 seeds 标为 incomplete；重复成功 seeds、不同代码版本或同一方法重复运行时混合环境均拒绝自动汇总。不同方法可以分别使用 CPU/CUDA 环境，但共有科学计算依赖必须一致，CSV 保存环境 fingerprint。全部108个 dataset/model 组合都有覆盖行，未运行或全部失败时分数为空。论文 reference 来自固定 commit 的 assets/results/f1_mean.csv 和 f1_std.csv。未做医疗数据的附加 fine-tuning 实验。

PyPOTS 1.5 可能在 training 遇到错误或中断后返回已有 checkpoint；runner 在 fit 期间监听该库的 error/interruption 记录，把这种 run 标记为 failed。正常 patience early stopping 仍可成功。回归测试故意在第二个 epoch 抛错，确认已有 checkpoint 不会让失败被算作成功。
