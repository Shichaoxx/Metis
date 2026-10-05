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
| 教学与 cookbook | 张量教程、训练/推理张量图、NFCorpus 数据准备与 ScoreHead 训练协议 | NFCorpus 是唯一完成真实训练的任务示例 |

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

## 待完成

通用 task/objective/backend 注册、单选/多选的完整评测与选优、概率校准、更多 backbone、稀疏 attention、RL、多卡、布局消融、多 seed 和第二个真实任务。Tree 仅有小规模分数、梯度与候选换序正确性预检，尚未验证训练质量或加速。相关实现与扩展方向见[项目对照](../research/related-projects.md)。
