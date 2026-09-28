# 现有原型设计与经验转化

本文解释 Metis 的现有代码（工程标识仍为 `cometa`），不代表通用后训练 Kit 已完成。产品要求以 [CORE](../project/core.md) 为准，验证范围与缺口见 [工作盘点](../project/status.md)。当前已接通 Qwen3 yes/no 与 Qwen base + ScoreHead；模型工厂用于训练和 Predictor 加载。真实微调执行状态见工作盘点。

设计来源：个人列排序工程经验，以及五道口纳什的视频和 [modern_genai_bilibili](https://github.com/wdkns/modern_genai_bilibili) 中关于 Jev / Tree Mask 的教学资料。具体来源、参考版本与独立实现范围见 [来源致谢](../../ACKNOWLEDGEMENTS.md)。框架组合与工程实现的价值，不等于相关 attention 方法的首创。

## 定位与任务边界

Agent 可按执行职责理解为：通用模型 + Harness + 任务能力与资源。环境、Memory、Skills、MCP 工具是不同层次的资源与接口，不需要硬塞进一个统一的模型层。Metis 位于任务能力层：输入业务状态与候选，输出分数、候选 ID、是否拒绝决定。它可以是本地函数、服务或工具，由 Harness 决定何时调用。

`SystemOne` 是这里给任务模型的产品定位，并非 Jev 官方架构的复现声明。最小任务可表示为 `f(query, context, candidates) -> scores`。ranking 用 BCE + pairwise softplus，multi_label 用 BCE，single_choice 用候选间 CE。现有接口不生成自然语言理由；它也不能独立完成长程计划、任意工具操作或环境交互。

## 从列排序经验抽取稳定设计

这部分将用户在 llm_ranker 上提出的经验转成框架要求；没有把企业代码、训练数据或内部评测复制进新项目。原项目详细复盘在个人学习仓库，本文只保留通用工程结论。

| 经验/问题 | 框架处理 | 收益 |
|---|---|---|
| 列本身有稳定业务身份 | 每个候选有唯一 ID | 调序、分块后仍能准确对齐分数 |
| 数据预处理与训练 prompt 容易漂移 | 训练/评测/推理复用 compiler | 输入模板、tokenizer 和截断可追溯 |
| JSONL 字段靠约定，缺失标注易变成负例 | schema_version、显式 labels、unjudged policy | 错误及早暴露；区分训练假设与判断事实 |
| 数据文件版本不明确 | manifest + 文件 SHA-256 + registry | 防止静默替换评测集 |
| 路径与产物靠脚本硬编码 | 配置相对路径、run ID、checkpoint/export分工 | 训练恢复与部署加载不互相混淆 |
| 日志只显示 loss，缺少谱系 | config、environment、事件、数据/模型版本 | 能解释一次实验用了什么 |
| decoder 串联候选造成方向性依赖 | pairs基线与tree分支隔离、统一逻辑位置 | 独立候选具有相同注意力可见性规则 |
| 调用方需要稳定的任务输出 | ScoreHead/yes-no 等读出统一转 scores、ID 和状态；当前已接通两种读出 | Harness直接消费结构化结果 |

这不是逐行代码移植。新代码从公共依赖和上述接口要求独立编写，列排序示例也只使用合成字段。

## 从原工程保留什么，改善什么

只读核查原列排序工程时，模型工厂已经提供 `score_head` 和 `yesno`，默认使用 `score_head`。ScoreHead 读取候选位置的 backbone hidden states 后经 MLP 得分，目标函数从外部注入；这些是值得保留的设计，不应因新原型选了预训练 reranker 而丢失。这里提炼接口经验，不迁移企业源码或数据。

| 原工程经验 | 在工具包中应保留或改善的方向 | 当前原型的位置 |
|---|---|---|
| 两种读出有明确模型工厂 | 基座、head/readout、loss 分开适配 | 已有 Qwen3 yes/no、Qwen base + ScoreHead，以及显式 model registry |
| 可训练的候选分数与外部损失 | 保留非生成监督任务，允许任务选择 BCE/排序/分类目标 | 当前已有固定目标分支，尚非可注册的 Objective 体系 |
| 列 ID 与候选读出位置 | 在训练、分块、评测、调用之间稳定对齐 | 已有 candidate ID、compiler 和输出对齐约定 |
| 数据、路径和输出依赖脚本约定 | 明确 schema、相对路径、run、checkpoint 与 export 层级 | 已有基础实现；不因此声称所有业务格式均已适配 |
| 业务系统调用排序模块 | 独立 Predictor/服务承载模型，Harness 组织业务流程 | 已有tiny合成示例及真实训练best的workflow调用；下游组装证据已验证，最终答案/业务收益未验证 |

## 输入语义与共享前缀

独立 pair 序列为 `prefix + candidate_i + readout`。编译器先按完整官方模板分词，再寻找安全公共 token 前缀，防止分别分词导致边界 token 不一致。tree 中前缀看自身过去，分支看前缀和自身过去，读出不能看兄弟分支。

各分支 position IDs 从公共前缀长度开始；“重新开始”指每个分支使用相同起点，不是无条件从 0 编号。树形排列的物理 offset 不能直接充当 RoPE 位置。超过物理长度限制时分块，每块都会重新计算前缀。

在 eval、相同输入且无 dropout 时，tree 与独立 pairs 应给出数值近似等价结果，候选置换只改变返回排列。训练时共享前缀的梯度自然累加；启用随机 dropout 后一次前向的随机图不要求逐元素相同，需区分结构正确性与随机训练轨迹。

## 薄框架与扩展顺序

核心数据、配置、注册、日志与 IR 指标用标准库；模型代码复用 Transformers、HF Trainer、PEFT。当前 DatasetRegistry 已接 CLI；Model Registry 已接通两个 Qwen3 适配器的训练/加载；通用 Registry 提供显式注册和重复名检查，task/objective/backend 尚未接通，不做自动插件发现。

增加新任务时先定义：候选是什么、标签语义、训练 loss、评测指标、推理决定及拒绝策略，然后实现适配器。schema linking 可沿用候选列与 ranking/multi_label；工具路由可用 single_choice/multi_label；有可定义动作与奖励的任务可考虑 RL，包括单步 contextual bandit 和多步交互；需要另行接入采样、反馈与优化目标，当前没有 RL trainer。

为了轻量，当前不重复实现分布式 launcher、GPU kernel 编译器、Trainer 优化器栈或完整 Agent runtime。相应功能缺口清楚列出，不能把接口草图写成已经支持。SFT 在这里是有监督任务后训练；当前不是通用聊天 SFT 数据格式。

## 核心不变量

- 每个输入候选对应且只对应一个有限数值分数，保留原始 ID。
- 标注、qrels、metadata 不进入模型输入。
- 验证/测试候选不按标准答案补全；没有召回到正例的 query 也参与完整指标。
- 显式 train-only 正例注入与弱负例有记录；不把 unknown 与 negative 隐式合并。
- 数据文件修改使哈希失效；注册表 pin manifest；产物复制后校验文件集合与内容。
- checkpoint 用于恢复优化进度；export用于推理。final 是末步状态；selection.enabled 时另外保存完整 dev nDCG@10 选出的 best。
- 评测失败的 query 不从总体均值悄悄剔除；整次评测标记失败。
- 当前 yes/no adapter 打分是 raw yes-minus-no logit；ScoreHead 返回 raw relevance logit。阈值与 calibration 需要任务验证，不默认宣称概率可靠。

## 性能路线

yes/no 适配器只投影两个词表行，ScoreHead 适配器只执行小型 MLP；两者均避免构造全词表 logits。当前采用 bounded pair batching 与 gradient checkpointing；query级loss仍保留全部候选的反传图，所以不能承诺峰值显存仅相当于一个小 batch。

第三步才是实际树形稀疏后端。PyTorch SDPA 接口支持不意味着特定 GPU Flash kernel 接受任意 mask；本项目不把 dense 4D mask包装成FlashAttention加速。可研究 FlexAttention 的 block mask 或把前缀/分支 softmax 的局部统计用 log-sum-exp 精确合并。需要同时验证前向、参数梯度、padding、GQA、位置、dtype、dropout和实际CUDA吞吐。

