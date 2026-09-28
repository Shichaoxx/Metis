# Metis 当前状态

Metis 是面向任务专用模型的 SystemOne 后训练工具包原型。包名和 CLI 仍为 cometa。首个完整案例使用 NFCorpus，工具包目标不限于检索排序。

## 已完成

- Qwen3-0.6B-Base + 独立 MLP ScoreHead + LoRA 的两轮监督训练，共 648 次优化更新。
- 按完整 dev 选择 best，重载导出产物，并通过 workflow 返回候选 ID、构造证据上下文。
- 首轮公开任务 test：本轮模型 nDCG@10 为 0.290978，原始 Qwen3-Reranker-0.6B 为 0.354783。方法、模板和训练历史不同，不能解释为单因素消融。
- pairs/SDPA 路径用于训练；tree 仅有正确性预检，没有稀疏加速证据。

## 仍未实现

通用 task/objective/backend 注册、稀疏算子、强化学习、多卡训练、多 seed 消融和第二个真实任务仍未完成。首轮模型质量低于所选强基线，当前没有 SOTA 结论。

实验使用 NFCorpus 与 Qwen 模型。数据和模型权重未随源码分发；请分别遵守其来源和许可条款。完整训练协议见 [ScoreHead cookbook](../../cookbooks/score-head-plan.md)。
