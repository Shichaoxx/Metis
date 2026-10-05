# 框架设计

Metis 将候选打分任务拆为数据、任务目标、输入编译、模型读出、训练和调用。当前实现包含 Qwen3 Base + ScoreHead 和 Qwen3 Reranker yes/no 两种适配器；通用 task/objective/backend 插件协议尚未实现。项目目标见[核心设计](../project/core.md)，验证范围见[项目状态](../project/status.md)。

候选打分接口统一排序、单选与多选任务。共享前缀与 Tree Mask 的教学参考见[来源致谢](../../ACKNOWLEDGEMENTS.md)。源码与合成示例为独立实现。

## 架构总览

以下图稿描述架构 v1.0：当前模型读出、候选可见性和评分契约保持固定。训练方法与执行后端的扩展另列于文末，不属于图中已实现路径。

![Metis toolkit architecture](figures/metis-architecture-en.png)

[英文 PDF](figures/metis-architecture-en.pdf) · [中文 PDF](figures/metis-architecture-zh.pdf) · [中文预览](figures/metis-architecture-zh.png) · [英文源文件](figures/metis-architecture-en.tex) · [中文源文件](figures/metis-architecture-zh.tex)

图中分别展示共享评分路径、监督训练和产物推理。两种模型适配器是替代选择，训练与推理复用相同的输入编译和评分接口。完整 dev 选优适用于 ranking；单选、多选已有损失与决策实现。英文和中文使用同一几何布局，仅替换标签。逐张量的维度与梯度另见 [ScoreHead](score-head.md)。

## 组件与接口

| 组件 | 职责 | 当前实现 |
|---|---|---|
| 数据与配置 | 固定输入、监督及实验身份 | JSONL schema、manifest、内容哈希、相对路径解析、DatasetRegistry |
| 任务目标与决策 | 定义监督损失及分数到选择结果的转换 | `tasks/objectives.py`、`tasks/decisions.py`；内置 ranking、multi_label、single_choice |
| 输入编译 | 统一训练与推理的 token、读出位置和截断规则 | pairs 与 dense tree；Qwen3 模板 |
| 模型与读出 | 将候选映射为标量分数 | Qwen3 Base + 独立 MLP ScoreHead；Qwen3 Reranker yes/no；显式模型工厂 |
| 训练与产物 | 优化、验证、恢复和导出 | HF Trainer、full/LoRA、独立 head 学习率、checkpoint、best/final export |
| 调用与决策 | 将分数转换为稳定的候选结果 | Predictor、候选 ID、TopK/单选/多选、开发用 HTTP 服务 |

数据、配置、注册、日志与 IR 指标使用标准库；模型和训练复用 Transformers、HF Trainer 与 PEFT。`Registry` 提供显式注册和重复名称检查，模型工厂已接入两个 Qwen3 适配器的训练与加载。task/objective/backend 仍是固定实现分支，没有通用注册协议或自动插件发现。

任务层与运行层分开维护：损失函数不负责 checkpoint 和模型加载，决策函数不依赖 Torch。`training.py` 调用任务目标，`api.Predictor` 调用任务决策；原有导入接口保持兼容。源码身份记录递归覆盖所有 Python 子模块。

## 任务与输出

当前任务接口为 `f(query, context, candidates) -> scores`，每个候选具有稳定 ID 和一个分数。

| 任务 | 监督目标 | 决策方式 |
|---|---|---|
| `ranking` | BCE + pairwise softplus | 按分数排序，可指定 TopK 或最低分数 |
| `multi_label` | BCE | 对各候选分数应用 sigmoid 与阈值 |
| `single_choice` | 候选间交叉熵 | 选择最高分候选 |

单选与多选沿用候选打分接口，没有独立的固定类别分类头。完整 IR 指标评测、训练候选采样和完整 dev 选优目前仅支持 `ranking`；单选、多选尚无真实任务 cookbook。sigmoid 输出的校准程度需要另行验证。

训练、评测和推理复用输入编译器。候选重排、分块和合并后，分数仍与原始 ID 对齐；labels、qrels 和 metadata 不进入模型文本。现有接口返回分数、候选 ID 和决策状态，不生成自然语言理由。

## 输入语义与共享前缀

独立 pair 序列为 `prefix + candidate_i + readout`。编译器先按完整模板分词，再寻找安全公共 token 前缀，避免分别分词改变边界 token。tree 中前缀可见自身过去，分支可见前缀和自身过去，读出不可见兄弟分支。

各分支的 position IDs 从公共前缀长度续编，使用相同起点。树形排列的物理 offset 与 RoPE 逻辑位置分别处理。超过物理长度限制时分块，每块重新计算前缀。

在 eval、相同输入且无 dropout 时，tree 与独立 pairs 应给出数值近似等价结果，候选置换只改变返回顺序。训练时共享前缀的梯度累加；启用 dropout 后，两种布局的单次随机前向不保证逐元素相同。

当前 tree 使用 dense mask，仅有小规模分数、梯度与换序正确性验证。首轮真实训练使用 pairs/SDPA，尚无 tree 训练质量或加速结果。

## 训练与产物约定

- 每个候选对应一个有限数值分数，输出保留原始 ID。
- 未判断标签与负例分开处理；train-only 正例注入和弱负例来源记录在数据中。
- 验证、测试候选不按标准答案补全；零正例召回 query 仍计入完整指标。
- manifest 记录数据哈希，注册表固定 manifest 身份；产物校验覆盖文件集合与内容。
- checkpoint 恢复优化器、调度器、随机状态等训练进度，export 用于推理。`final` 保存末步状态；启用 selection 时，`best` 保存完整 dev nDCG@10 最优产物。
- query 评测失败时，整次评测标记失败，不以成功子集替代完整结果。
- yes/no 读出为 raw yes-minus-no logit，ScoreHead 返回 raw relevance logit；阈值与概率校准按任务验证。

## v1.0 的完善方向

以下工作保持现有模型结构和候选打分接口，尚未完成的项目不构成当前能力声明。

| 优先级 | 工作 | 验证方式 |
|---|---|---|
| 1 | 为单选、多选补齐评测、dev 选优和真实 cookbook | 单选 accuracy / macro-F1，多选 micro/macro-F1；固定候选与划分，独立报告零召回情况 |
| 2 | 改善训练候选与监督质量 | 审计弱负例与未判断样本；固定候选召回后比较困难负例、损失权重和采样方案 |
| 3 | 校准分数与拒答阈值 | 在独立校准数据或明确划分的 dev 子集拟合温度及阈值，报告 NLL、Brier、校准误差与覆盖率/错误率 |
| 4 | 测量并优化执行开销 | 按候选数、前缀/分支长度报告端到端延迟、吞吐和峰值显存，分别统计编译、前向与决策 |

评测指标随任务语义选择；排序分数不能未经验证就作为概率。所有超参数、校准与阈值选择只使用训练/验证数据，固定后再评估 test。新增实验记录新的配置和数据身份，保留已有 cookbook 的冻结结果。

## 强化学习接入设计

本节是后续训练扩展设计，当前没有 RL trainer。现有 `f(query, context, candidates) -> scores` 可以作为候选策略的打分器，无需改变 Qwen3 Base + ScoreHead 的前向结构。

最小接入可从单步单选任务开始：对候选分数 `s` 计算 `softmax(s / temperature)`，按分布采样一个候选，执行后取得奖励，再用动作的 log-probability 构造策略梯度损失。这属于 contextual bandit：一次决策对应一次反馈；多步任务还需定义状态转移、终止条件与回报归因。[PyTorch 的 REINFORCE 示例](https://docs.pytorch.org/docs/stable/distributions.html#score-function)说明了采样与 log-probability 如何参与梯度估计。

新增职责应位于评分模型之外：

| 扩展职责 | 最小契约 |
|---|---|
| 候选策略 | scores 到动作分布；采样、有效候选 mask、动作 log-probability、推理决策 |
| 反馈采集 | 输入及候选身份、实际动作、行为策略版本与采样概率、奖励、失败状态 |
| 奖励定义 | 按任务计算成功率、质量或成本；固定可核验的评测环境 |
| 优化与评测 | 策略梯度目标、降低方差的基线、可选监督约束，以及独立的任务成功率评测 |

LoRA 路径可继续只更新 adapter 与 ScoreHead。用于降低方差的基线不必引入新的可训练 value head，因此最小方案不要求改变现有模型结构。推理仍可采用确定性 argmax/TopK；训练采样与部署决策需要分别评测。

排序任务的动作是有序候选列表，需要定义无放回采样及整个列表的 log-probability，例如 Plackett–Luce 策略；不能将确定性 TopK 当作已采样动作。[Neural PG-RANK](https://arxiv.org/abs/2310.04407)提供了评分模型到排序策略的研究参考，其实验结果不能直接视为 Metis 的效果。离线历史日志的训练还需要行为策略概率和候选覆盖等条件，不能将监督排序损失或缺少采样记录的点击日志直接称为已实现的策略梯度训练。

verl 的生成式 rollout、token log-probability 与 Metis 的候选动作接口不同。接入此类训练框架需要适配动作和反馈契约，仅添加 reward 函数不足以完成接入。当前优先保留单进程、单步候选策略的设计边界，不新增分布式 RL 依赖。

## 其他扩展方向

新增任务需要定义候选、标签、损失、指标与决策策略。schema linking 可使用候选列的排序或多选，工具路由可使用单选或多选。RL 还需要动作、采样、反馈及优化目标，当前没有 RL trainer。

Metis 模型可作为函数或服务接入 Agent workflow，通用 LLM 与 Harness 负责规划、生成、环境交互和流程状态。工具包不包含完整 Agent runtime，也未实现分布式训练 launcher。

yes/no 适配器只投影两个词表行，ScoreHead 只执行小型 MLP，均避免构造全词表 logits。当前训练使用 pair microbatch 与 gradient checkpointing；query 级损失仍保留全部候选的反向图，峰值显存受整条 query 的候选数影响。

稀疏 tree 后端是待研究方向，可考虑 FlexAttention block mask，或用 log-sum-exp 合并前缀与分支的局部 softmax 统计。SDPA 接口本身不保证任意 dense mask 使用 Flash kernel；后端实现需要分别验证前向、梯度、padding、GQA、位置、dtype、dropout 和 CUDA 吞吐。

## 设计参考

[LlamaFactory 的框架设计](https://aclanthology.org/2024.acl-demos.38/)以模型加载、数据处理和训练模块划分职责；[HybridFlow / verl](https://arxiv.org/abs/2409.19256)将流程组织与模型执行解耦。Metis 借鉴这些边界划分，当前仍采用单进程 HF Trainer 与固定任务分支。

论文图参考上述论文的职责和数据流表达，并使用 [ML Architecture Diagram](https://github.com/Ztsdut/ml-architecture-diagram-skill/tree/94b074e8f19de8b1be199343470fbd9a2b1bb3b4) 的结构核验与排版规则，由本项目独立绘制。图稿使用本地 TikZ 渲染；绘图工具不属于运行库依赖。
