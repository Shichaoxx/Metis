# Metis 当前状态

## 组件与实现状态

Python 包、import 和 CLI 均为 `metis`。

| 组件 | 当前实现 | 验证范围与待完善项 |
|---|---|---|
| 数据与任务 | JSONL schema、manifest/哈希、配置、DatasetRegistry；ranking、multi_label、single_choice | 单选、多选使用候选打分，没有独立分类头；尚无对应真实任务 cookbook |
| 输入与 attention | 公共输入编译器、pairs/dense tree、逻辑位置、eager/SDPA | tree 已验证小规模分数、梯度与换序一致性；没有稀疏加速或真实训练结果 |
| 模型与读出 | Qwen3 Base + MLP ScoreHead、Qwen3 Reranker yes/no、模型工厂 | 当前仅有两个 Qwen3 适配器 |
| 监督训练 | HF Trainer、full/LoRA、独立 head 学习率、混合精度、候选采样 | 候选采样仅支持 ranking；尚无 RL 或多卡训练 |
| 评测与选优 | IR 指标、完整 dev 选优、原始模型/训练产物评测及报告比较 | 完整指标评测与 dev 选优仅支持 ranking |
| 实验与产物 | 日志、checkpoint 恢复、best/final 导出、完整性校验 | LoRA 产物依赖记录的基座，恢复 checkpoint 依赖其历史选优目录 |
| 推理与调用 | Predictor、候选分数与 ID、TopK/单选/多选、HTTP、workflow 示例 | HTTP 为开发服务；workflow 已验证证据上下文组装，尚未评估最终答案 |
| 教学与 cookbook | 张量教程、训练/推理张量图、NFCorpus 训练协议、BoolQ 读出研究 | NFCorpus 完成工具包完整闭环；BoolQ 使用独立研究入口，未接入通用任务选优与 Predictor |

已有通用 Registry 容器、DatasetRegistry 和模型工厂；内置任务损失与决策集中在 `tasks/`，与 Trainer 和 Predictor 分离。task/objective/backend 插件协议尚未实现。项目提供核心、CPU 模型和分发三组 CI 配置，验证方法见[贡献指南](../../CONTRIBUTING.md)。

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

该配置下，ScoreHead 的 nDCG@10 低于原始 reranker。两者的输入模板、读出和训练历史不同，结果用于完整方法的对照。评测固定 BM25 top50，保留零正例召回 query；弱负例和训练正例注入的处理见 [cookbook](../../cookbooks/score-head-plan.md)。

Best/final 分开导出；重载 150 个候选的最大分数差为 0。Workflow 使用模型排序结果构建证据上下文，尚未评估下游 LLM 答案质量。

## BoolQ 读出结构研究

2026-10-06 在真实 BoolQ 标签上完成冻结 Qwen3-0.6B-Base 表示的四种评分头、三个种子实验。固定 384 train / 96 dev / 192 held-out evaluation，各组 360 次优化更新，按 dev NLL 选模型；12 个选定头的新进程重载分数完全一致。

冻结表示阶段，MLP 平均 accuracy 为 63.02%，SwiGLU 为 63.37%，SwiGLU + 标量残差为 64.76%，训练多数类参考为 65.63%。残差相对 MLP 的探索性差值区间包含零。单种子联合 LoRA 阶段完成三组、共 216 次优化更新，accuracy 分别为 61.46%、59.38%、55.21%；没有支持替换默认读出的稳定收益。该研究使用独立保存格式，完整配置与结果见 [BoolQ 消融](../../cookbooks/boolq-readout-ablation.md)。

2026-10-07 完成固定 MLP 的第二轮读出实验：1,536 train / 192 dev / 384 新预留 evaluation，比较末尾 Relevance 标记、Decision 标记与候选正文注意力池化，每种读出使用三个训练种子。每组 3 epochs、288 次优化器调用，共九组；最后一次调用学习率为零。按各组 dev NLL 选 checkpoint，九组全部冻结后才评测新的 evaluation。

| 读出 | 平均 Dev NLL | Evaluation accuracy | 平衡准确率 | 平均 NLL | 平均 Brier |
|---|---:|---:|---:|---:|---:|
| 训练先验常数参考 | — | 58.33% | 50.00% | 0.6844 | 0.2455 |
| 末尾 Relevance | 0.6799 | 58.33% | 50.00% | 0.6818 | 0.2443 |
| 末尾 Decision | 0.6811 | 58.33% | 50.00% | 0.6824 | 0.2446 |
| 候选注意力池化 | 0.6827 | 58.33% | 50.00% | 0.6833 | 0.2449 |

开发集选出的读出仍为 Relevance。九组在固定 0.5 阈值下均预测正类；Decision 和池化相对 Relevance 的 NLL/Brier 配对区间包含零，没有显示分类收益。相同决策不能证明架构等价，微小概率指标变化也不构成任务能力提升。

全部选定 head/LoRA 均有实际更新，池化 query 也有更新，冻结基座保持不变；九组新进程重载的三条 dev 参考分数差均为零。峰值已分配显存约 2.40 GiB。该实验使用独立保存格式，不是生产 `Predictor` 产物；Decision 是现有词表中的文本标记，不是新增可学习 token 或层内注入。协议、置信区间和输入审计见 [BoolQ 读出位置与池化](../../cookbooks/boolq-readout-selection.md)。

两轮研究不改变 v1 模型、冻结 NFCorpus 配方或原实验产物，也不计为 RL 或完整 BoolQ 基准。开源决策模型常用任务与训练方式见 [Jev 风格任务调研](../research/jev-task-survey.md)。

## 待完成

通用 task/objective/backend 注册、单选/多选的完整评测与选优、概率校准、更多 backbone、稀疏 attention、RL、多卡、真实训练布局消融，以及第二个任务的工具包完整闭环。Tree 仅有小规模分数、梯度与候选换序正确性预检，尚未验证训练质量或加速。BoolQ 已有多种子联合 LoRA 研究，仍需建立超越简单基线的任务质量。相关实现与扩展方向见[项目对照](../research/related-projects.md)。
