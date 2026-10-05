# Metis 当前状态

## 已有能力

- Qwen3 Base + 独立 MLP ScoreHead，以及 reranker yes/no 读出。
- Full fine-tuning / LoRA、独立 head 学习率、混合精度、候选采样与完整 dev 选优。
- 数据和配置契约、日志、checkpoint 恢复、best/final 导出与完整性校验。
- Pairs / dense tree 输入、eager / SDPA；Predictor、结构化候选结果与开发 HTTP 服务。
- Python 包、import 和 CLI 统一为 `metis`，只维护一套当前版本和 `main` 分支。

## NFCorpus 首轮训练

Qwen3-0.6B-Base + ScoreHead + LoRA 已完成两轮监督排序训练，共 **648 次参数更新**。本轮使用 pairs/SDPA，有 2,556,417 个可训练参数；head 和 LoRA 均有实际更新。

| 选优阶段 | Dev nDCG@10 |
|---|---:|
| 随机 head 初始值，仅作参照 | 0.090762 |
| Step 324 | 0.242403 |
| Step 648，选定 best | 0.263831 |

完整 dev 为 324 query；选定 best 后完成一次 323-query test、新进程重载和实际产物的 workflow 调用。

| 模型 | Test nDCG@10 |
|---|---:|
| 原始 Qwen3-Reranker-0.6B | 0.354783 |
| Base + ScoreHead + LoRA | 0.290978 |

这轮 ScoreHead 落后于 reranker。两者的输入模板、读出和训练历史不同，结果用于当前方案对照。固定 BM25 top50，保留零正例召回 query；弱负例和训练正例注入的处理见 [cookbook](../../cookbooks/score-head-plan.md)。

Best/final 分开导出；重载 150 个候选的最大分数差为 0。Workflow 使用模型排序结果构建证据上下文，尚未评估下游 LLM 答案质量。

## 待完成

通用 task/objective/backend 注册、更多 backbone、稀疏 attention、RL、多卡、布局消融、多 seed 和第二个真实任务。Tree 仅有小规模分数、梯度与候选换序正确性预检，尚未验证训练质量或加速。
