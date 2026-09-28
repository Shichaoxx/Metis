# 参考资料与证据边界

核查日期：2026-09-27。下列链接支持方法或接口说明，不代表 Metis 已实现上游的全部功能。`main` 文档可能继续变化；实际复现实验应另存模型 revision、库版本和代码 commit。

## R0. 五道口纳什的视频与 GitHub 教学资料

本项目明确参考 Bilibili 创作者 **[五道口纳什](https://space.bilibili.com/59807853)** 的 Jev / Tree Mask 视频讲解，以及 [wdkns/modern_genai_bilibili](https://github.com/wdkns/modern_genai_bilibili) 中的 [jev_architecture-v2.pdf](https://github.com/wdkns/modern_genai_bilibili/blob/6b8729b923d17863014ac6e0ff7715fc84473300/slides/jev_architecture-v2.pdf)。GitHub URL 与版本由本地 Git 验证；远端在线状态和具体视频 BV 链接本次尚未核实。

对共享状态、候选分支可见性和张量化教学表达的参考应保留署名。本项目重新构造例子与图形、独立编写实现，不将借鉴思想或已有方法称为首创。该课件对 Jev 的架构分析不等于 Jev 官方披露。原课件和视频不随本项目分发；本地参考版本未发现明确许可证，署名不替代授权。完整记录见 [来源与版权说明](../../ACKNOWLEDGEMENTS.md)。

## R1. Qwen3-Reranker 的原始读出

- [Qwen3-Embedding 官方仓库：Reranker Transformers Usage](https://github.com/QwenLM/Qwen3-Embedding#qwen3-reranker)
- [可直接检查的 README 源码](https://github.com/QwenLM/Qwen3-Embedding/blob/main/README.md)
- [Qwen3-Reranker-0.6B 模型页](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)

官方示例使用固定 prefix/suffix，在最后一个有效输入位置提取 `yes`、`no` token 的 logits，对二者做 softmax。教程中的 `s=z_yes-z_no` 与这个二分类概率的 log-odds 对应。它不等同于在所有候选之间做 softmax，也不自动具备可靠概率校准。

## R2. RoPE 与逻辑位置

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

参考的是旋转位置编码通过内积体现相对位置的性质。教程中分支从共享前缀长度续编的位置方案，是为匹配独立 pair 输入而采用的结构；仅正确设置 mask 不足以保证这种匹配。

## R3. Attention 接口与 mask 约定

- [PyTorch scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch 实现及函数文档](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py)
- [PyTorch FlexAttention](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)

SDPA 布尔 mask 的 True 表示可参与 attention。具体 kernel 选择依赖设备、dtype、mask 等条件。支持某个张量接口，不等于该输入一定使用稀疏 kernel。

## R4. 共享前缀与归一化合并

- [Hydragen: High-Throughput LLM Inference with Shared Prefixes](https://arxiv.org/abs/2402.05099)
- [Flash Preference](https://github.com/li-plus/flash-preference)
- [DPO Prefix Sharing](https://github.com/frankxwang/dpo-prefix-sharing)

Hydragen 是共享前缀推理的重要参考。教程中的 logsumexp 合并公式直接由 softmax 分区推导，不能将两部分各自归一化后的输出简单相加。Flash Preference 和 DPO Prefix Sharing 是共享前缀训练的实现参考；其支持模型和任务各有限制，并不证明 Qwen3-Reranker 能即插即用。教程不报告它们的速度数字作为本项目结果。

## R5. 训练框架与 adapter 保存

- [Transformers Trainer customization](https://github.com/huggingface/transformers/blob/main/docs/source/en/trainer_customize.md)
- [SentenceTransformers CrossEncoder 训练概览](https://github.com/huggingface/sentence-transformers/blob/main/docs/cross_encoder/training_overview.md)
- [SentenceTransformers RankNetLoss](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/cross_encoder/losses/rank_net.py)
- [PEFT 自定义模型与 modules_to_save](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/custom_models.md)
- [PEFT checkpoint 格式](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/checkpoint.md)

Metis 使用 Trainer 作为通用生命周期基础，把候选数据、目标函数和导出约定留在任务层。非生成打分、pairwise 排序损失、LoRA 本身都已有成熟实现，不应被表述为本项目的新算法。主配方保留 yes/no vocabulary readout；后续若新增 task head，则必须同时保存其参数。

## R6. NFCorpus、BEIR 与评测协议

- [NFCorpus 原始项目与使用条款](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/#terms-of-use)
- [BEIR 仓库与 dataset table](https://github.com/beir-cellar/beir)
- [BEIR 原始 README](https://github.com/beir-cellar/beir/blob/main/README.md)
- [BEIR 格式 NFCorpus 公共归档](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip)
- [BEIR 论文](https://arxiv.org/abs/2104.08663)

使用原始 train/dev/test，不随机混合后重新划分。BM25 候选召回率决定 reranker 的上限；评估完整 qrels 中未被召回的正例，不能只在已保留正例上计算召回分母。训练弱负例和注入正例的策略由具体 cookbook/manifest 明示。原作者条款允许学术用途免费使用；其他用途需取得原作者许可，不能称为通用商用开源授权。BEIR 代码许可不等于所有数据集的许可。

## 与 Jev 资料的关系

五道口纳什的视频及上述 GitHub 课件是本次讨论与教程的重要参考，涉及共享 state、树状可见性、位置重编号和结构化读出。本教程重新构造了 token 例子、图解与操作路线，没有转载原课件页面或复用其中实验数字。Jev 的黑盒行为不能唯一确定其内部网络、训练目标或 kernel；本文仅解释一个公开可实现、可测试的方案。

## 如何复查本教程

1. 初版 PDF 与生成器已归档到 `docs/history/tensor-guide-v1/`；生成器保留当时接口，仅用于重建旧版教程。当前讲解以 [张量教程](../learning/tensors.md) 和 [ScoreHead](../architecture/score-head.md) 为准。
2. 用 attention/model/training 测试核查 pairs/tree 对齐、梯度与导出契约。
3. 用 NFCorpus cookbook 运行质量和性能评测；任何实际结果应带数据与模型版本，独立于教学示意数字报告。
