# BoolQ 读出结构消融

本实验比较任务专用评分头，使用真实 BoolQ 标签和 Qwen3-0.6B-Base。它是探索性小样本实验，不是完整 BoolQ benchmark，也不改变 Metis v1 的默认架构。

## 固定协议

| 项目 | 设置 |
|---|---|
| 数据 | `google/boolq`，revision `35b264d03638db9f4ce671b711558bf7ff0f80d5` |
| 训练 / 开发 | 从官方 train 中固定选取 384 / 96 条，passage 分组隔离 |
| 最终评测 | 官方 validation 的固定 192 条；称为 held-out evaluation，不称官方 test |
| 采样 | seed `20261006`，按输入哈希排序；训练与开发排除所有官方 validation passage |
| 基座 | Qwen3-0.6B-Base，28 层，hidden size 1024；本地权重文件哈希保存于协议 |
| 输入 | question 作为 query，passage 作为唯一 candidate；instruction 明确判断肯定答案是否得到支持 |
| 编译 | 当前 ScoreHeadCompiler，pairs / SDPA，max length 256，候选尾部截断，保留问题 |
| 读出 | 最后一个 relevance suffix token 的 hidden state |
| 标签与目标 | BoolQ answer；单标量 BCE，sigmoid 表示肯定概率；标签不进入 token 输入 |
| 指标 | accuracy、balanced accuracy、binary Brier、NLL；10-bin ECE 作为小样本辅助指标 |
| 模型选择 | 每种结构单独用开发集 NLL 选择训练后的 checkpoint；最终评测不参与选优 |

基座输入相同，门控头使用 SiLU 门控投影与值投影的逐元素乘积；残差头增加从基座表示直接到标量的线性路径。第一轮不加入 Norm，避免把门控、归一化与残差混为一项改动。新增线性路径采用常规随机初始化，不是 JevAny 的零初始化残差修正。相同种子控制随机数与样本顺序，但不意味着不同结构的初始函数或分数尺度相同。

| 结构 | 宽度 | 参数量 |
|---|---:|---:|
| 线性参考 | — | 1,025 |
| GELU MLP，与 v1 同结构 | 256 | 262,657 |
| SwiGLU 评分头 | 128 | 262,529 |
| SwiGLU + 标量残差 | 128 | 263,554 |

后三种头的参数差异小于 0.4%。线性头用于检验复杂头是否必要，不是等参数比较。门控设计参考 [GLU Variants](https://arxiv.org/abs/2002.05202)，效果需要在此任务上实际验证。

## 两阶段验证

**阶段 A：冻结表示。** 基座处于 eval 模式且参数冻结，一次性提取固定特征。四种头均从头训练，使用相同缓存、epoch 内样本顺序与优化设置；种子为 42、43、44。AdamW，学习率 0.001，weight decay 0.01，batch size 32，30 epochs。不做输入归一化、温度调优或 RL。训练集多数类与类先验概率作为参考。

**阶段 B：联合 LoRA。** 比较 MLP、SwiGLU、SwiGLU + 残差；固定 seed 42、3 epochs，LoRA r8 / alpha16 / dropout0，作用于 q/k/v/o。基座 LoRA 学习率 0.00002，头学习率 0.001，weight decay 0.01，microbatch 2，梯度累积 8，梯度范数裁剪 1.0，无学习率调度。参数采用 FP32、计算采用 BF16 autocast。三个架构的训练预算相同，每个头 72 次优化更新。单种子联合训练不能估计完整训练随机性的影响。

阶段 A 只回答固定表示上的读出能力；阶段 B 才检验联合适配。两者均不验证专用读出 token、注意力池化、RL 或完整数据集泛化。

阶段 A 与 B 的优化预算、头部计算精度、随机数消耗及头初始化不同，不能把跨阶段差值解释为 LoRA 的单因素收益；架构比较应在各阶段内部进行。

## 复现入口

准备项目的 `train`、`peft` 可选依赖与 `pyarrow`。下载并固定 Qwen3-0.6B-Base 权重后执行：

```bash
python examples/research/extract_boolq_features.py \
  --model /path/to/Qwen3-0.6B-Base \
  --data data/boolq --output runs/boolq/features --device cuda:0

python examples/research/train_readout_ablation.py \
  --features runs/boolq/features/features.safetensors \
  --protocol runs/boolq/features/protocol.json \
  --output runs/boolq/frozen-heads --mlp-width 256

python examples/research/train_boolq_lora_ablation.py \
  --pilot runs/boolq/features --model /path/to/Qwen3-0.6B-Base \
  --output runs/boolq/joint-lora --device cuda:0 --precision bf16

python examples/research/train_boolq_lora_ablation.py \
  --verify runs/boolq/joint-lora/mlp/seed-42/best --device cuda:0
```

运行目录保存原始数据哈希、模型身份、固定样本索引、特征缓存、每轮开发指标、实际参数更新、选定模型与预测。研究头的保存格式独立于 v1 Predictor 产物，不可混用。BoolQ 遵守其 [CC-BY-SA-3.0 数据条款](https://huggingface.co/datasets/google/boolq)；本仓库不分发数据或模型权重。

## 结果

### 阶段 A：冻结基座，三种子

2026-10-06 完成 12 组真实训练，每组 360 次优化更新；每组仅根据开发集 NLL 选定模型。下表为三个种子的平均结果，accuracy 标注样本标准差。

| 评分头 | Accuracy ↑ | Balanced accuracy ↑ | NLL ↓ | Binary Brier ↓ |
|---|---:|---:|---:|---:|
| 训练先验 / 多数类参考 | 65.63% | 50.00% | 0.6437 | 0.2257 |
| 线性 | 63.02% ± 2.27 pp | 51.98% | 0.6470 | 0.2280 |
| MLP | 63.02% ± 1.88 pp | 52.47% | 0.6426 | 0.2257 |
| SwiGLU | 63.37% ± 3.05 pp | 50.20% | 0.6557 | 0.2316 |
| SwiGLU + 残差 | 64.76% ± 1.31 pp | 56.55% | 0.6434 | 0.2262 |

残差门控相对 MLP 的平均准确率差为 +1.74 pp，配对 bootstrap 的探索性 95% 区间为 **[-1.39, +5.03] pp**，包含零。重采样单位为评测样本，先对固定的三个训练种子求样本正确率均值；该区间不包含完整的数据与训练随机性，也未进行多重比较校正。

192 条评测记录涉及 191 个 passage，存在一个重复 passage 组；上述区间未按 passage 聚类，只作探索性描述。后续[读出位置实验](boolq-readout-selection.md)使用每个 passage 仅一条记录的新评测子集。

所有头的平均准确率均未超过多数类参考；门控的 NLL/Brier 也未改善。此阶段不支持替换默认 MLP，亦不能证明残差无效。模型的 balanced accuracy 仅略高于随机水平，提示固定表示加少量标签在此设置下不足以稳定解决任务。

12 个选定头在独立进程重载后，对缓存评测表示的 logits 与原记录完全一致。特征抽取采用共享 RTX 3090，BF16、batch4；峰值 allocated 约 1.15 GiB。训练 / 开发 / 评测分别有 52/384、6/96、30/192 条候选发生尾部截断。头部 CPU 延迟仅描述评分头计算，不作为完整模型的延迟证据。

### 阶段 B：联合 LoRA

三个架构均完成 3 epochs、72 次优化更新，共 216 次更新；使用同一 LoRA 初始化和逐 epoch 样本顺序。开发集 NLL 分别选中 MLP 的 epoch1、SwiGLU 的 epoch1、残差门控的 epoch3，再各评测一次固定的 192 条样本。

| 评分头 | Accuracy ↑ | Balanced accuracy ↑ | NLL ↓ | Binary Brier ↓ |
|---|---:|---:|---:|---:|
| 训练先验 / 多数类参考 | 65.63% | 50.00% | 0.6437 | 0.2257 |
| MLP | 61.46% | 57.65% | 0.6779 | 0.2423 |
| SwiGLU | 59.38% | 54.26% | 0.6770 | 0.2420 |
| SwiGLU + 残差 | 55.21% | 44.95% | 0.6999 | 0.2515 |

SwiGLU 相对 MLP 的 accuracy 差为 -2.08 pp，探索性配对 95% 区间为 [-10.42, +5.73] pp；残差门控为 -6.25 pp，区间 [-13.54, +1.56] pp。两个区间均包含零，不能据此证明结构普遍更差。此轮没有观察到稳定收益，也不足以支持更换默认读出。

LoRA 与头均有实际更新，冻结基座的参数哈希保持不变。三个训练的 GPU 峰值 allocated 均约 2.38 GiB；共享 GPU 的运行时间不作为架构速度对照。该小样本、单种子、固定超参数试验没有超过多数类准确率参考；MLP 的 balanced accuracy 提升不等于所有质量指标提升。

三个 best 产物均已在新进程加载基座、LoRA 与头；每个产物的 3 条开发参考样本 logits 与导出时完全一致。完整 192 条评测使用选定产物重载后的对象执行，不重测留出集来调参。18 项离线 CPU 契约测试通过，覆盖数据隔离、有限梯度、实际参数更新、开发集选优及保存加载；这些软件测试与真实任务质量分开记录。

该实验只验证普通 MLP、SwiGLU 与随机初始化标量 skip 的候选方案。新增可学习 token、零初始化残差修正与层内注入仍未验证。末尾文本标记和候选池化的多种子联合训练见[第二轮研究](boolq-readout-selection.md)，使用新的预留评测子集，不根据本轮最终评测继续调参。
