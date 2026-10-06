# Jev 风格开源模型：任务、数据与训练

核查日期：2026-10-06。以下项目为独立实现；[TypeSafe Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python) 是服务客户端，不提供 Jev 的训练源码或权重。项目公布的结果尚未在 Metis 中复现。

## 常见任务

| 任务 | 典型数据 | 输出与评价 |
|---|---|---|
| 意图、工单与工具路由 | Banking77、MASSIVE、合成 workflow | 单选；accuracy、macro-F1、拒答与执行成功率 |
| 证据判断与自然语言推断 | BoolQ、MNLI、ChaosNLI | 二元概率或三分类分布；accuracy、NLL、Brier |
| 规则与政策判断 | 程序生成规则、合成政策与场景 | 二元判断或动作单选；规则泛化与边界案例 |
| 等级评分 | SST5、合成评分 rubric | 等级分布；accuracy、期望等级误差与概率质量 |
| 答案核验与知识单选 | GSM8K-Verify、MMLU-Pro | 正误概率或答案分布；accuracy、校准与推理开销 |
| 候选动作与排序 | Wikispeedia、游戏状态、偏好数据 | 下一步动作或排列；成功率、排序指标与交互成本 |

同一个任务可以采用判别头、答案 token 或生成式推理实现。输出接口相似，并不意味着训练方法或计算量相同。

## 代表实现

| 项目与核查版本 | 数据与模型 | 训练与结果证据 |
|---|---|---|
| [Kev](https://github.com/jaredpalmer/kev/tree/5e42a7a03f28134853dd3ff77461457e921e5ec1) | Banking77、BoolQ、AG News、MNLI、SST5 与规则数据；Qwen Base + LoRA + pointer head | CE，另做温度校准；有版本化模型卡、数据身份与评测输出。结构与 Metis 的独立读出方向相近 |
| [Hmm](https://github.com/n4ze3m/hmm/tree/78d8d4a62a41857ce67d903aadf504580feea832) | 公共数据与 DeepSeek 合成 workflow；Qwen3.5-4B + LoRA | 用原 LM 头的选项字母做硬/软标签 CE；作者报告 typed-decisions 从未微调 0.596 到 0.709，说明合成标签未经人工审查，不是 RL 结果 |
| [Tev1](https://github.com/togethercomputer/tev1/tree/1dde7782382c9f49d627153759b8d1deab426ce0) | MNLI、BoolQ、Banking77、AG News、SST5 与合成任务；37,840 train、4,568 dev；Qwen3.5-4B + LoRA | 单答案字母 SFT；开发 benchmark 有结果文件，README 明确不属于未触碰的最终测试 |
| [Laya](https://github.com/NandhaKishorM/laya/tree/a4a8921afebfd852bba0000475cfb6ab737a124c) | NLI、新闻、情感、意图与合成 workflow；encoder + 两层 Transformer 头 + option scorer | 包含 proper-score、Gaussian-logit exploration 与 CE；域内微调质量声明的部分结果缺少已提交报告，不能当作已复现收益 |
| [OpenJev-RLCD](https://github.com/ZimmyGao/openjev-rlcd/tree/22db40ad3333f5cf0bc5139097da70e0f8c05018) | GSM8K 答案核验、MMLU-Pro、ChaosNLI；Qwen3-1.7B | 采样生成推理文本，读取答案 token 概率；proper-score 与 REINFORCE；[预印本](https://arxiv.org/abs/2609.38850)、三种子与结果日志。生成成本与无生成评分头不同 |
| [JevAny](https://github.com/SimpleJev/JevAny/tree/33cb677bbe300ad853f2d2a061f2e1ea34e7fcba) | 分类、政策、工具/动作、偏好/排序等；LoRA + pointer 或 direct-token | 发布权重采用 CE/SFT，RLCR 属于实验扩展；发布 corpus 不能由公开 builder 完整重建。[数据边界](https://github.com/SimpleJev/JevAny/blob/33cb677bbe300ad853f2d2a061f2e1ea34e7fcba/docs/DATA.md) |
| [Jevlike](https://github.com/vinnylarouge/jevlike/tree/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452) | 合成菜单、Wikispeedia、游戏 imitation；byte encoder 或冻结 HF encoder | option attention scorer；合成高分与真实下一步点击结果差距较大，适合作为学习与对照实现 |

以上结果使用不同数据、划分、模型规模、提示与指标，不能按数值直接排列模型优劣。CE、单答案 token SFT、带推理采样的 RL 与独立判别头应分别归类。

JevAny 的 [pointer 算法](https://github.com/SimpleJev/JevAny/blob/33cb677bbe300ad853f2d2a061f2e1ea34e7fcba/docs/ALGORITHM.md) 包含点积评分与非线性残差修正。其 [技术报告 §5.3](https://github.com/SimpleJev/JevAny/blob/33cb677bbe300ad853f2d2a061f2e1ea34e7fcba/reports/JevAny_Tech_Report.pdf) 提供 250-step、904 个开发决策的头部对照：

| JevAny 评分头 | 参数量 | Accuracy ↑ | 校准后 NLL ↓ | Brier ↓ |
|---|---:|---:|---:|---:|
| MLP scorer | 2,688,001 | 82.41% | 0.4935 | 0.2596 |
| Residual pointer | 2,688,001 | 82.63% | 0.4586 | 0.2497 |

这是作者提供的短程单种子筛选，表中未报告差值区间。pointer 以决策表示与候选表示交互，残差输出层零初始化；报告未定义该对照 MLP 的具体输入，不能直接等同 Metis 的单 hidden-state MLP。此对照可支持实验设计，不能把 +0.22 pp 的开发集差值当作普适或已复现收益。

## 读出与可见性差异

Metis 的候选分支相互隔离，单个 raw score 只依赖公共 query/context 与自身候选。Kev/JevAny 默认隔离的是 question 分支；同一 question 内放置全部 options 和末尾 decision token，后面的 option 可读取前面的 option，decision token 可读取全部 options。[Kev 输入与 mask](https://github.com/jaredpalmer/kev/blob/5e42a7a03f28134853dd3ff77461457e921e5ec1/kev/model.py)、[JevAny 输入与 mask](https://github.com/SimpleJev/JevAny/blob/33cb677bbe300ad853f2d2a061f2e1ea34e7fcba/jevany/model.py)。

两类模型均支持动态候选数，但 pointer 的 decision 表示可依赖整个候选集合。直接移植 pointer 模板会同时改变读出、输入布局与候选可见性，不能把差异全部归因到评分头。独立候选打分与联合候选选择应作为两种任务条件分别验证。

Kev/JevAny 可配置 `option_isolation=True` 来隔离 option spans；末尾 decision token 仍汇聚全部 options，因此并不等同 Metis 的候选独立评分契约。

参数预算也需匹配。带 bias 的 scalar MLP 为 `(D+2)d+1`，双投影 pointer 为 `2(D+1)d_p`。`D=1024` 时，MLP 宽 256 有 262,657 个参数，pointer 宽 256 则有 524,800；pointer 宽 128 才接近当前 MLP 的预算。

## Metis 的实验选择

首先在 [BoolQ](https://huggingface.co/datasets/google/boolq) 上进行小规模二元判断。每条自然问题配一段证据文本，只需一个标量肯定概率，适合控制计算预算并比较读出结构。[原始论文](https://arxiv.org/abs/1905.10044)。具体协议和实测结果见 [BoolQ 读出消融](../../cookbooks/boolq-readout-ablation.md)。

后续可采用三类互补任务：

1. [MNLI](https://cims.nyu.edu/~sbowman/multinli/)：entailment、neutral、contradiction 三候选单选。公开带标签的 matched/mismatched 是 validation，不能称为官方 test。
2. [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets)：原作者 10,003 train、3,080 test、77 intents；训练集内划分 dev。完整 77-way 与加入正确标签的少候选试验是不同协议。
3. SST5 等级评分与现有 NFCorpus 排序：分别检验等级输出和候选排序，不能只用分类准确率代替所有任务指标。

架构实验先固定数据、基座、读出位置、损失、更新预算和模型选择规则；RL 另作训练方法实验。数据、权重与原始日志在本地保存，源码仓库只提供协议、脚本与汇总结果。
