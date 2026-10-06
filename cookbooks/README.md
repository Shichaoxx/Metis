# Cookbook

当前示例使用 NFCorpus：

- [数据准备](nfcorpus.md)：来源、split、BM25 top50 和候选标签。
- [ScoreHead 训练](score-head-plan.md)：已完成的训练、dev 选优、最终评测、导出与 workflow 调用。

新任务需明确输入、标签、损失、指标、固定 split 和调用方式。配置放在 [recipes](../recipes/README.md)，数据、权重与运行报告单独保存。

架构实验：[BoolQ 读出结构消融](boolq-readout-ablation.md)，分别验证冻结表示的评分头与联合 LoRA 训练。

[BoolQ 读出位置与池化](boolq-readout-selection.md)固定同一 MLP，比较末尾标记与候选正文注意力池化；三个训练种子使用新的预留评测子集。
