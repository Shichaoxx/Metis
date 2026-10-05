# Cookbook

当前示例使用 NFCorpus：

- [数据准备](nfcorpus.md)：来源、split、BM25 top50 和候选标签。
- [ScoreHead 训练](score-head-plan.md)：已完成的训练、dev 选优、最终评测、导出与 workflow 调用。

新任务需明确输入、标签、损失、指标、固定 split 和调用方式。配置放在 [recipes](../recipes/README.md)，数据、权重与运行报告单独保存。
