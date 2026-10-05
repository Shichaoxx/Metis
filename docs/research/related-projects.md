# 相关项目与设计对照

核查日期：**2026-10-05**。比较依据为官方仓库的默认分支、README 和关键训练、推理源码；提交日期使用 Asia/Shanghai。下列实现能力经过源码核对，模型质量与性能数字未在 Metis 环境复现。

Metis 面向专用任务模型的后训练、评测、导出与调用。当前已有能力以候选打分为中心，完整真实训练验证来自 NFCorpus；具体组件和限制见[项目状态](../project/status.md)。

## Jev 与可训练的本地决策模型

TypeSafe 将 Jev 定义为输入状态、输出带概率的结构化决策的 System One 模型，官方介绍提出 RLCD 训练方法。[官方介绍](https://typesafe.ai/blog/introducing-system-one-models-and-jev)和 [Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)可用于理解产品与接口；SDK 是服务客户端，不能视作 Jev 的模型权重或训练实现。社区项目中的同名 RLCD 也不因此等同于官方训练算法。

| 项目 | 已有实现 | 与 Metis 的关系 |
|---|---|---|
| [Kev](https://github.com/jaredpalmer/kev) | Qwen backbone + pointer head；Noul/Choice/Score；LoRA 与全参训练、温度校准、公开权重、本地服务 | 自行训练决策模型的直接参照。其读出对 question/option 向量打分，Metis 当前使用候选末位置的独立 MLP |
| [AgentJev](https://github.com/malevrigns/agent-jev) | Qwen3-0.6B 无 LM head；候选末 token 读出、候选集合 Transformer 和标量打分；Boolean/Choice/Score；训练与公开权重 | 与 Metis 当前 backbone 和读出位置最接近；它进一步建模候选集合，而 Metis 当前 pairs/tree 路径保持候选相互隔离 |
| [Laya](https://github.com/NandhaKishorM/laya) | Encoder + decision head；typed decisions、checkpoint router、本地推理；训练模块、soft-CE/RLCD 目标、温度校准与导出 | 任务接口、校准与 Agent 接入的参考；采用 encoder 路线，当前仓库也提供训练代码 |
| [SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev) | 冻结开放模型，提取答案槽位的词表 logits；选项分布、多后端与温度校准 | 不更新 backbone 的基线参照；在核查版本中未发现通用权重微调入口。旧 SemIf 地址重定向到同一仓库 |

上述四个项目均包含本地模型推理实现。将输入包装后转发至 Jev API 的 SDK、MCP 服务或 Agent skill，则属于另一类接入工具。

### 核查版本与源码入口

| 项目 | 默认分支最新提交 | 模型实现 | 训练或推理入口 |
|---|---|---|---|
| Kev | 2026-10-04 · [fe64b12](https://github.com/jaredpalmer/kev/commit/fe64b1274ea7f80d4095866df90666abb03e9cf6) | [PointerHead](https://github.com/jaredpalmer/kev/blob/fe64b1274ea7f80d4095866df90666abb03e9cf6/kev/model.py) | [训练](https://github.com/jaredpalmer/kev/blob/fe64b1274ea7f80d4095866df90666abb03e9cf6/kev/train.py) |
| AgentJev | 2026-10-01 · [1c2c1b1](https://github.com/malevrigns/agent-jev/commit/1c2c1b1ae1dc427d4cc851ef7460a1112b0cb3e1) | [候选读出与集合建模](https://github.com/malevrigns/agent-jev/blob/1c2c1b1ae1dc427d4cc851ef7460a1112b0cb3e1/agentjev/model.py) | [训练](https://github.com/malevrigns/agent-jev/blob/1c2c1b1ae1dc427d4cc851ef7460a1112b0cb3e1/agentjev/train.py) |
| Laya | 2026-10-05 · [8a6e132](https://github.com/NandhaKishorM/laya/commit/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2) | [模块目录](https://github.com/NandhaKishorM/laya/tree/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2/laya) | [训练、校准与导出](https://github.com/NandhaKishorM/laya/blob/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2/laya/train.py) |
| SemIf | 2026-09-24 · [23cf1f3](https://github.com/TheoLeeCJ/SemIf-OpenJev/commit/23cf1f39fc9534fe81437200959b6dfc7106e45a) | [源码目录](https://github.com/TheoLeeCJ/SemIf-OpenJev/tree/23cf1f39fc9534fe81437200959b6dfc7106e45a/src/semif_phase1) | [直接读取 logits](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/23cf1f39fc9534fe81437200959b6dfc7106e45a/src/semif_phase1/direct.py) |

发布权重可在作者的 [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b/tree/main)、[AgentJev](https://huggingface.co/aimeigaoshou/agent-jev/tree/main) 和 [Laya](https://huggingface.co/convaiinnovations/laya/tree/main) 模型仓库中查看。复现时需固定权重 revision；代码 commit 与模型 revision 是不同的身份。

## 相邻训练框架

| 项目 | 主要交集 | 适合参考的组件 |
|---|---|---|
| [Sentence Transformers / CrossEncoder](https://github.com/huggingface/sentence-transformers/blob/4a3b5cd6ec718e421f57e824a41ed3fd99595df6/docs/cross_encoder/training_overview.md) | 成对文本打分、排序与分类训练 | Trainer、可组合损失、evaluator、多数据集训练；与 Metis 当前排序工程最直接重叠 |
| [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/blob/fd1a2bdf69488ffebe0327999d4400d8c8058a0b/examples/finetune/reranker/README.md) | Encoder/decoder reranker 微调 | 正负候选格式、知识蒸馏、reranker 适配和加载 |
| [LlamaFactory](https://github.com/hiyouga/LlamaFactory/tree/ce9dc9e072f80fa3abe0989d4ab90da25f083438) | 配置化后训练与模型适配 | 配置、训练生命周期和适配器组织；其主要目标是通用 LLM/VLM 微调 |
| [SetFit](https://github.com/huggingface/setfit/tree/be332d6d4993e272fd9bca4364e8025880228b48) | 少样本专用分类 | 数据规模较小时的分类基线与训练接口；采用 Sentence Transformer 路线 |

这些项目已覆盖非生成打分、专用任务训练和本地调用等能力。Metis 的设计空间在于将任务定义、backbone/readout/layout、目标函数、评测和产物契约组织为可复用的后训练工具包，并提供对应代码的张量说明与真实任务配方。通用 Task/Objective/Backend 扩展接口目前仍待完成，不能将这一目标描述为已有能力。

## 对实现路线的启示

1. **任务与评测接口优先。** 已有 ranking、multi_label、single_choice 的输入、损失和决策分支；需要补齐单选、多选的指标与 dev 选优，并抽取 Task/Objective/Evaluator/Decision 契约。
2. **用第二个真实任务检验复用。** 列选择或分类 cookbook 应复用训练、导出和 Predictor，并验证数据契约与决策语义，无需先扩大排序 benchmark 数量。
3. **区分分数、概率与决策。** 原始 logits、归一化选项分布、独立二分类概率和校准置信度有不同含义。Metis 目前提供分数与选择结果，尚无完整概率校准和校准评测组件。
4. **分别验证候选交互与执行效率。** 集合建模会改变任务条件；共享前缀和稀疏算子影响执行。Metis 的 dense tree 仅有小规模正确性验证，仍需独立评测质量、吞吐与显存。

上游 README 中的速度和质量属于作者报告。不同数据、模型规模、训练历史、硬件和指标下的结果，不能直接与 Metis 的 NFCorpus nDCG@10 排序。Laya 所称 RLCD 在核查版本中包含 noisy-logit sampling 的 GRPO 风格项与 soft CE，也不等于已实现环境交互式 RL。
