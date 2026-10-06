# BoolQ：末尾读出与候选注意力池化

本实验使用 Qwen3-0.6B-Base、真实 BoolQ 标签和联合 LoRA 训练，比较三种任务表示的读出方式。评分头保持两层 GELU MLP，实验变量是表示的汇聚位置与规则。协议于 2026-10-07 冻结；九组训练、开发集选取冻结、新进程重载和预留评测均已完成。三种读出在固定阈值下均未超过训练多数类参考，Metis v1 默认读出保持不变。

这是第二轮探索性实验。第一轮 [评分头消融](boolq-readout-ablation.md) 的配置和结果已经观察过；本轮使用新的预留评测样本，但不是官方 BoolQ test，也不是整个研究过程中完全未触碰的通用 benchmark。本实验不修改 Metis v1 默认模型或其产物格式。

## 数据与任务

[BoolQ](https://huggingface.co/datasets/google/boolq) 将自然语言问题与证据段落配对，标签表示问题的肯定答案是否成立。[原始论文](https://arxiv.org/abs/1905.10044)。数据 revision 固定为 `35b264d03638db9f4ce671b711558bf7ff0f80d5`；原始 parquet、选取索引、处理源码和模型文件的 SHA256 保存在运行协议中。

每条记录将 question 作为 query、passage 作为唯一 candidate，输出一个标量 logit。训练使用 binary cross-entropy with logits；推理取 sigmoid，概率语义为 `P(affirmative answer | question, passage)`。标签不进入模型文本。

| 划分 | 条数 | 来源与约束 |
|---|---:|---|
| 训练 | 1,536 | 官方 train；可复用第一轮训练记录 |
| 开发 | 192 | 官方 train；保留第一轮 96 个开发 passage，再按输入哈希补齐 |
| 本轮预留评测 | 384 | 官方 validation 的新子集；排除第一轮 192 条评测记录涉及的全部 191 个 passage |

split seed 为 `20261007`。选取顺序由 question/passage 输入哈希及 split seed 决定，不依赖标签；各 split 内只保留每个 passage 的一条记录。训练和开发排除**全部官方 validation passage**，并相互按 passage 隔离。第一轮开发 passage 永不进入本轮训练。新评测记录与旧评测按 passage 隔离，不只排除旧记录 ID。

`train_dev_records.json` 仅包含训练和开发记录；`sealed_evaluation.json` 单独保存评测记录。`selected_records.json` 保存全部 split 的来源索引与输入哈希，不包含问题、段落或标签。各文件的哈希登记在 `protocol.json`，训练 worker 不读取封存评测文件。

## 三种读出

三种方案共用 Qwen3-0.6B-Base、同一 instruction、MLP 宽度 256，以及最长 256 token 的输入限制。输入采用 pairs / SDPA，一次前向处理两个独立因果序列；不同样本之间不共享注意力。

| 方案 | 表示来源 | 额外可训练参数 |
|---|---|---:|
| `last_relevance` | 原 relevance 后缀的最后一个有效 token | 0 |
| `decision_marker` | 段落末尾文本 `\n<Decision>:` 的最后一个有效 token | 0 |
| `candidate_attention` | 对保留下来的候选正文 token 做可训练 query 注意力池化 | 1,024 |

`decision_marker` 使用 tokenizer 已有词表编码普通文本，没有新增词表 token 或独立可训练 embedding。因此，这项实验验证的是末尾读出标记，不是层内 token 注入、prefix tuning 或新的 embedding 架构。

三个方案统一预留 relevance 和 decision 两种后缀中较长者所需的 token 预算，再编译候选正文。这样，更换标记不会改变保留下来的 passage 内容。候选正文仍按尾部截断；各方案的正文 token 哈希、截断数和序列长度保存在输入审计中。

对于候选注意力池化，令隐藏状态为 $H\in\mathbb{R}^{B\times L\times D}$，$D=1024$，可训练向量为 $q\in\mathbb{R}^{D}$：

\[
e_{bt}=\frac{q^\top H_{bt}}{\sqrt{D}},\qquad
\alpha_{bt}=\operatorname{softmax}_{t\in\mathcal C_b}(e_{bt}),\qquad
h_b=\sum_{t\in\mathcal C_b}\alpha_{bt}H_{bt}.
\]

集合 $\mathcal C_b$ 只包含保留正文中带有 candidate 字符的 token。request-only prefix、读出后缀及 padding 都不参与池化；候选 hidden state 仍通过因果注意力读取此前问题与 instruction。若 tokenizer 将分隔符与首个候选字符合并成一个不可拆分 token，该 token 被纳入候选集合，并单独统计 `candidate_boundary_merge_tokens`。注意力能量、softmax 与加权求和使用 FP32；q 初始化为零，对应正文 token 上的均匀注意力。

三种表示最终经过相同结构：

\[
h\in\mathbb{R}^{1024}
\rightarrow\operatorname{Linear}(1024,256)
\rightarrow\operatorname{GELU}
\rightarrow\operatorname{Linear}(256,1)
\rightarrow z.
\]

MLP 有 262,657 个参数，池化方案合计增加 1,024 个 query 参数。本轮不是严格等参数比较，应同时报告这项差异。每个 seed 下 MLP 初始权重相同；输出层 weight 为零，bias 为训练集肯定比例的 logit，使三个方案具有相同的初始常数预测。LoRA 初始化与逐 epoch 样本顺序也按相同 seed 控制，并在选取冻结时核对。

## 训练与模型选择

| 项目 | 固定设置 |
|---|---|
| 训练组合 | 三种读出 × seeds `42 / 43 / 44`，共九组 |
| 训练长度 | 3 epochs；每组 288 次 optimizer 调用 |
| 批处理 | 实际 microbatch 2，累积 8 次，effective batch 16 |
| 精度 | FP32 参数与 optimizer 状态，BF16 autocast 计算 |
| LoRA | r8、alpha16、dropout0；作用于 q/k/v/o，基座原始权重冻结 |
| 优化器 | AdamW；LoRA learning rate `2e-5`，MLP 与 pool query learning rate `2e-4` |
| 正则与调度 | weight decay `0.01`；前 10% optimizer 调用 warmup，之后 cosine decay |
| 裁剪 | 全部可训练参数的梯度范数上限 `1.0` |
| checkpoint | 每个 epoch 导出一次；只在训练后的三个 checkpoint 中按开发集 NLL 选 best |
| 概率后处理 | 不拟合温度、不调整阈值；分类阈值固定为 sigmoid `0.5` |

学习率在最后一次 cosine 调度调用为零，因此 288 表示 optimizer 调用预算，不能全部称为非零参数更新。训练审计另行验证 LoRA、MLP 及适用时的 pool query 确实变化，冻结基座哈希保持一致。

所有九个 best 必须先完成开发集选择，再由 `finalize` 写入 `selection-lock.json`。冻结检查覆盖协议、选取记录、训练审计、best 产物哈希，以及相同 seed 下的公共初始化、样本顺序和正文 token 身份。**在完整选取锁生成之前，不打开本轮 384 条评测记录。** 评测只加载各组已选定产物，不根据评测结果重新选择 epoch、温度、阈值或结构。

指标包括 accuracy、balanced accuracy、binary NLL、binary Brier 和 10-bin ECE；小样本 ECE 作为辅助指标。训练多数类与先验概率作为参考，先验仅从训练标签估计。结果按三种读出分别汇总三个 seed，报告均值、样本标准差以及适用的配对差值；评测样本重采样与完整训练随机性应分别说明。

## 复现入口

安装项目的 `train`、`peft` 可选依赖及 `pyarrow`，准备固定的 Qwen3-0.6B-Base 本地权重，并保留第一轮的 `protocol.json` 和 `selected_records.json`。下面使用通用示例目录；运行数据与权重不随源码分发。

注册新的数据和九组训练计划：

```bash
python examples/research/prepare_boolq_readout_protocol.py \
  --data data/boolq \
  --model /path/to/Qwen3-0.6B-Base \
  --previous-pilot runs/boolq/features \
  --output runs/boolq/readout-selection
```

启动各读出方案，每次命令执行该方案的三个已登记 seed：

```bash
for readout in last_relevance decision_marker candidate_attention; do
  python examples/research/train_boolq_readout.py train \
    --protocol runs/boolq/readout-selection \
    --model /path/to/Qwen3-0.6B-Base \
    --readout "$readout" --device cuda:0 --precision bf16
done
```

九组训练全部完成后，先冻结选择，再执行各组评测：

```bash
python examples/research/train_boolq_readout.py finalize \
  --protocol runs/boolq/readout-selection

for readout in last_relevance decision_marker candidate_attention; do
  python examples/research/train_boolq_readout.py evaluate \
    --protocol runs/boolq/readout-selection \
    --model /path/to/Qwen3-0.6B-Base \
    --readout "$readout" --device cuda:0
done
```

GPU 显存预算可通过 `--gpu-memory-fraction` 配置。共享设备上的训练时长只用于记录资源使用，不作为三种读出的受控速度比较。

全部评测完成后，只读取已保存报告进行核验与汇总：

```bash
python examples/research/summarize_boolq_readout.py \
  --protocol runs/boolq/readout-selection
```

汇总输出三种子的均值与标准差、训练先验参考，以及相对 `last_relevance` 的 accuracy/NLL/Brier 配对 bootstrap；不加载模型、不再运行推理。

## 产物与重载

每组训练保存完整开发指标、checkpoint、best 选择与参数变化审计。best 是独立研究产物，包含 LoRA、MLP、可选 pool query、tokenizer、编译与读出配置、三条固定开发参考输入及其 logits，并带文件完整性清单。

评测阶段在新进程中加载选定产物，先校验文件哈希与开发参考 logits，再处理新预留评测记录。研究格式为 `metis-research-boolq-readout-v2`，不能作为 Metis v1 `Predictor` 的导出模型加载。BoolQ 遵守其原始 CC-BY-SA-3.0 条款；仓库不分发数据、模型权重或原始运行报告。

## 结果

2026-10-07：九组真实训练与完整评测已完成。`last_relevance` 按三个已选 checkpoint 的平均开发集 NLL 选为本轮候选；预留评测指标不参与选择。下表报告三个 seed 的均值 ± 样本标准差。Accuracy 与 balanced accuracy 使用百分比，标准差使用百分点；其余指标保持原始尺度。训练先验参考只有一个固定预测器，肯定概率为 `0.6328125`。

| 读出 | Dev NLL ↓ | Accuracy ↑ | Balanced accuracy ↑ | NLL ↓ | Binary Brier ↓ | ECE ↓ |
|---|---|---|---|---|---|---|
| 训练先验 / 多数类参考 | — | 58.33% | 50.00% | 0.6844 | 0.2455 | 0.0495 |
| `last_relevance` | 0.6799 ± 0.0014 | 58.33% ± 0.00 pp | 50.00% ± 0.00 pp | 0.6818 ± 0.0017 | 0.2443 ± 0.0008 | 0.0400 ± 0.0126 |
| `decision_marker` | 0.6811 ± 0.0018 | 58.33% ± 0.00 pp | 50.00% ± 0.00 pp | 0.6824 ± 0.0021 | 0.2446 ± 0.0010 | 0.0396 ± 0.0127 |
| `candidate_attention` | 0.6827 ± 0.0027 | 58.33% ± 0.00 pp | 50.00% ± 0.00 pp | 0.6833 ± 0.0029 | 0.2449 ± 0.0013 | 0.0392 ± 0.0176 |

预留评测集有 224 个肯定标签和 160 个否定标签。九个模型在阈值 `0.5` 下均将全部 384 条样本预测为肯定，因而 accuracy 与多数类参考完全相同，balanced accuracy 为 50%。NLL 和 Brier 的均值略低于训练先验参考，但未转化为分类收益；较小的 ECE 也不能证明模型具备有效的判别能力。

相对 `last_relevance` 的配对差值定义为候选方案减去参考方案。Accuracy 以百分点表示，正值更好；NLL 和 Brier 保持原始尺度，负值更好。

| 候选方案 | Accuracy Δ [95% interval] | NLL Δ [95% interval] | Brier Δ [95% interval] |
|---|---|---|---|
| `decision_marker` | 0.00 [0.00, 0.00] pp | +0.0006 [−0.0007, +0.0018] | +0.0003 [−0.0003, +0.0009] |
| `candidate_attention` | 0.00 [0.00, 0.00] pp | +0.0015 [−0.0033, +0.0065] | +0.0006 [−0.0016, +0.0029] |

区间由 10,000 次配对 bootstrap 计算，seed 为 `20261007`。先对相同样本的三个固定训练 seed 的指标差值取平均，再以唯一评测 passage 为单位重采样。区间仅描述评测抽样的不确定性，不包含重新训练的随机性；结果为探索性比较，未作多重比较校正。三个训练 seed 的样本标准差单独报告。

两个候选方案相对 `last_relevance` 的 NLL/Brier 差值区间均包含零，未显示改善。Accuracy 区间退化为零只表示本轮九组分类决策一致，不能证明架构等价或确定总体泛化误差。该结果支持在本轮条件下保留默认末尾读出，不能推出末尾标记或注意力池化在其他数据、训练预算和任务上始终无效。

## 训练与输入审计

每组完成 288 次 optimizer 调用，九组合计 2,592 次；各组末次调用的学习率为零。训练审计验证九个已选模型的 LoRA 和 MLP 权重均已变化，三个池化模型的 query 也已变化，冻结基座的参数哈希保持不变。九次新进程重载均通过三条固定开发参考输入的核验，最大 logit 绝对差为 `0.0`；这项核验不增加预留评测次数。

| 读出 | Seeds | 各 seed 的 best epoch | MLP / pool / LoRA 参数 | 总可训练参数 | 峰值分配显存 GiB |
|---|---|---|---:|---:|---:|
| `last_relevance` | 42 / 43 / 44 | 3 / 3 / 1 | 262,657 / 0 / 2,293,760 | 2,556,417 | 2.400 |
| `decision_marker` | 42 / 43 / 44 | 3 / 3 / 1 | 262,657 / 0 / 2,293,760 | 2,556,417 | 2.399–2.400 |
| `candidate_attention` | 42 / 43 / 44 | 3 / 3 / 1 | 262,657 / 1,024 / 2,293,760 | 2,557,441 | 2.400–2.401 |

显存为单个训练进程记录的 PyTorch peak allocated memory，不代表整卡占用，也不包含其他进程。训练运行于共享 GPU；单组耗时约 980–1,026 秒，不能据此作受控吞吐或结构加速结论。

| 读出 | 划分 | 条数 | 被截断条数 | 总输入 tokens | 平均 / 最长 tokens |
|---|---|---:|---:|---:|---:|
| `last_relevance` | train | 1,536 | 184 | 260,406 | 169.54 / 256 |
| `last_relevance` | dev | 192 | 16 | 33,016 | 171.96 / 256 |
| `decision_marker` | train | 1,536 | 184 | 258,870 | 168.54 / 255 |
| `decision_marker` | dev | 192 | 16 | 32,824 | 170.96 / 255 |
| `candidate_attention` | train | 1,536 | 184 | 260,406 | 169.54 / 256 |
| `candidate_attention` | dev | 192 | 16 | 33,016 | 171.96 / 256 |

三种读出的正文 token 身份及截断条数一致。`decision_marker` 的最终序列较短一个 token，差异来自读出后缀，不改变保留正文。上表仅覆盖 train/dev，不推断预留评测的 token 数量。

协议 SHA256：`ebcfa42e102fe95d8dd88b84046edf6191060bfb5427cde9a2ccf5f6e02cc54e`。

选择锁 SHA256：`936cd8e0a651f0a69071e4546b5535efb2318155ec6d933624705e9496f1aac5`。

本轮比较应在三个读出之间进行。相较第一轮，训练数据量、开发划分、初始化控制、head learning rate、学习率调度和 seed 数均发生变化；跨轮差值不能单独归因到训练先验初始化、某个读出或单一架构模块。这些结果属于 BoolQ 上的第二轮探索性读出实验，不构成 Metis v1 架构的普遍性能结论。
