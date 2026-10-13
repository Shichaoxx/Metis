# 开放决策模型：任务、数据与训练

核查日期：2026-10-07。覆盖 Jev 风格实现、任务专用蒸馏工具包、动态标签分类模型和相关推理服务。各项分别说明源码、训练入口、权重与数据的公开情况；项目公布的质量和速度尚未在 Metis 中复现。

## 检索范围与证据

检索词包括 `Jev`、`OpenJev`、`System One`、`decision model`、`calibrated decisions`、`dynamic label classification` 和 `task distillation`，并沿模型卡、论文与上游引用补查。社区目录用于发现候选，技术判断依据作者仓库、训练源码、模型卡和论文。表中的 GitHub 链接固定到核查提交；可变的模型下载入口不代表与该代码天然匹配，复现时还需固定模型和数据 revision。

这是一份有范围的资料核查，不是全网穷尽清单，也不是复现排行榜。仓库公开不等于训练数据、发布配方和权重都公开；GitHub 的非 fork 标记也不证明代码没有共同祖源。同名项目按所有者区分，派生模型、服务封装和同一项目的不同权重不计作独立架构。

[TypeSafe Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python) 是服务客户端，不提供 Jev 的训练源码或权重。社区项目的 RLCD 实现不等于官方训练方法的完整复现。

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

## 已有主要参照

| 项目与核查版本 | 数据与模型 | 训练与结果证据 |
|---|---|---|
| [Kev](https://github.com/jaredpalmer/kev/tree/5e42a7a03f28134853dd3ff77461457e921e5ec1) | Banking77、BoolQ、AG News、MNLI、SST5 与规则数据；Qwen Base + LoRA + pointer head | CE，另做温度校准；有版本化模型卡、数据身份与评测输出。结构与 Metis 的独立读出方向相近 |
| [Hmm](https://github.com/n4ze3m/hmm/tree/78d8d4a62a41857ce67d903aadf504580feea832) | 公共数据与 DeepSeek 合成 workflow；Qwen3.5-4B + LoRA | 用原 LM 头的选项字母做硬/软标签 CE；作者报告 typed-decisions 从未微调 0.596 到 0.709，说明合成标签未经人工审查，不是 RL 结果 |
| [Tev1](https://github.com/togethercomputer/tev1/tree/1dde7782382c9f49d627153759b8d1deab426ce0) | MNLI、BoolQ、Banking77、AG News、SST5 与合成任务；37,840 train、4,568 dev；Qwen3.5-4B + LoRA | 单答案字母 SFT；开发 benchmark 有结果文件，README 明确不属于未触碰的最终测试 |
| [Laya](https://github.com/NandhaKishorM/laya/tree/a4a8921afebfd852bba0000475cfb6ab737a124c) | NLI、新闻、情感、意图与合成 workflow；encoder + 两层 Transformer 头 + option scorer | 包含 proper-score、Gaussian-logit exploration 与 CE；域内微调质量声明的部分结果缺少已提交报告，不能当作已复现收益 |
| [OpenJev-RLCD](https://github.com/ZimmyGao/openjev-rlcd/tree/22db40ad3333f5cf0bc5139097da70e0f8c05018) | GSM8K 答案核验、MMLU-Pro、ChaosNLI；Qwen3-1.7B | 采样生成推理文本，读取答案 token 概率；proper-score 与 REINFORCE；[预印本](https://arxiv.org/abs/2609.38850)、三种子与结果日志。生成成本与无生成评分头不同 |
| [JevAny](https://github.com/SimpleJev/JevAny/tree/2b9ed4c20f0e3d86db28bf7ddc1c11ac80c6dfe0) | 分类、政策、工具/动作、偏好/排序等；LoRA + pointer 或 direct-token | 发布权重采用 CE/SFT，RLCR 属于实验扩展；发布 corpus 不能由公开 builder 完整重建。[数据边界](https://github.com/SimpleJev/JevAny/blob/2b9ed4c20f0e3d86db28bf7ddc1c11ac80c6dfe0/docs/DATA.md) |
| [Jevlike](https://github.com/vinnylarouge/jevlike/tree/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452) | 合成菜单、Wikispeedia、游戏 imitation；byte encoder 或冻结 HF encoder | option attention scorer；合成高分与真实下一步点击结果差距较大，适合作为学习与对照实现 |

以上结果使用不同数据、划分、模型规模、提示与指标，不能按数值直接排列模型优劣。CE、单答案 token SFT、带推理采样的 RL 与独立判别头应分别归类。

## 基座与任务头、专用任务训练

这些项目与 Metis 的模型或工具包设计直接相关，既包括 decoder，也包括 encoder。可训练接口和发布模型的训练配方是两项不同的证据。

| 项目与核查版本 | 架构与任务 | 已公开材料及限制 |
|---|---|---|
| [JevForge · 6cf0dfe](https://github.com/zwliJay/jev-forge/tree/6cf0dfe3b811d623331c0cc8b1c00ff13614772e) | Qwen 0.6B/0.8B；每个候选独立成行，末 token → 两层 GELU 标量头 → 组内 softmax；Mind2Web 页面动作、判断和等级评分 | [模型源码](https://github.com/zwliJay/jev-forge/blob/6cf0dfe3b811d623331c0cc8b1c00ff13614772e/jevforge/model.py)、训练/校准/RL 入口、[0.8B 权重](https://huggingface.co/AndeyTait/JevForge-0.8B)、[数据](https://huggingface.co/datasets/AndeyTait/JevForge-Mind2Web)与固定划分结果。与 Metis pairs + MLP 很接近；未复现其质量或效率 |
| [Strands Decider · 3e94e9d](https://github.com/strands-labs/strands-decider/tree/3e94e9d84c620ed5a95f1a3310c3decb971e261c) | Qwen3.5-2B-Base + LoRA + pointer head；Choice/Noul/Score | [训练配方](https://github.com/strands-labs/strands-decider/blob/3e94e9d84c620ed5a95f1a3310c3decb971e261c/training/README.md)、数据构建、校准、[权重](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21)与多次重训报告。监督训练并包含 teacher KL；同一问题内联合读取选项，与独立候选评分有别 |
| [NanoJev · 76fdfc9](https://github.com/TianyuCodings/NanoJev/tree/76fdfc9ecdca45a9bcef17991a07d3041a87685a) | Qwen3-0.6B + 共享标量头；Choice 增加候选集合 attention 残差；程序概率题和游戏动作 | 全参监督训练、[权重](https://huggingface.co/C-Tianyu/NanoJev)、[游戏数据](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data)及 episode 结果公开。发布游戏模型为专家监督训练，不能把仓库中的其他 RL 实验归给该权重；不同游戏有明显成败差异 |
| [Luce · bda3d21](https://github.com/scienthoon/luce/tree/bda3d21cdb537ae99662fdba275adc450e90f455) | 默认 Qwen3-4B-Base + LoRA；多种独立头/读出；任务定义 → teacher 合成或标注 → 训练 → 评测 → 服务 | 有训练代码、公共数据转换和任务报告，包括多数类坍缩负结果。[已核实的公开 adapter + head](https://huggingface.co/noscienthoon/ouro-2.6b-decision-lora)对应另一项 Ouro-2.6B 实验，不能当作 Qwen4B 任务报告的权重 |
| [AgentJev · 1c2c1b1](https://github.com/malevrigns/agent-jev/tree/1c2c1b1ae1dc427d4cc851ef7460a1112b0cb3e1) | Qwen3-0.6B；候选末 token → 集合 Transformer → 残差与 scalar MLP；soft CE + Brier | [typed v1 权重](https://huggingface.co/aimeigaoshou/agent-jev/tree/7d433994fbde17a3f0993c2f2b02fe8ca1370db1)与协议公开。HF main 已改为编码完成判断；其新 RL 训练入口引用的部分模块未在核查提交找到，不能称整套训练可自足重建。候选集合层会改变 Metis 的评分独立性 |
| [firelex/jeff · 7d47c42](https://github.com/firelex/jeff/tree/7d47c42a9255cc7070af52cc0d0a49fce90d3795) | Qwen3.5-0.8B + 独立 255 槽线性头，以原 LM head 答案代码行初始化；硬/软标签 CE | 训练器、数据来源、[adapter 权重](https://huggingface.co/mstrasser/jeff-base)公开。当前 v1.3 的完整训练混合未发布，脚本还保留旧版本配方；固定答案槽位不等同于候选共享标量头 |
| [S1LV3RJ1NX/openjev · 91847b4](https://github.com/S1LV3RJ1NX/openjev/tree/91847b4c678f7d9330d8f3475e24c72c7fef73c2) | Qwen3-1.7B LoRA 与 ModernBERT 两条路径；选项 marker 读出、共享状态、多问题输入；任务专用微调 | [训练器](https://github.com/S1LV3RJ1NX/openjev/blob/91847b4c678f7d9330d8f3475e24c72c7fef73c2/scripts/train.py)、通用/专用权重、279 任务混合及路由数据公开。区分未见任务零样本与有监督任务适配，报告少数类失败与迁移负结果；其路由模型和零样本 Jev 比较是非对称训练条件 |
| [GLiClass · 68132de](https://github.com/Knowledgator/GLiClass/tree/68132def761c2dccc20d1a71c04abdc5db35463f) | 动态标签分类框架；支持多种 encoder 布局、pooling、点积/MLP scorer、单标签与多标签 | [论文](https://arxiv.org/abs/2508.07662)、训练器、微调示例和公开模型；另有分类策略 RL 入口。工具包抽象与任务定义值得参考，不能只因不叫 Jev 而排除 |
| [GLiFormer · 19db82a](https://github.com/Knowledgator/GLiFormer/tree/19db82a339e5b5ce35e4ea3dc7db4e3912422d76) | DeBERTa encoder 与动态标签头；分类、判断和评分 | 作者模型代码与[权重](https://huggingface.co/knowledgator/gliformer-large-v1)公开，是下表另一 Jeff 服务所加载的模型。服务封装和底层模型应分开归类 |
| [Shalimov04/open-jev · 6126df4](https://github.com/Shalimov04/open-jev/tree/6126df4b8d9163af835048476a7eee0858b81dbb2) | teacher 概率 → 小型 encoder 固定标签分类器；KL 蒸馏，可加 gold CE；标签、训练、校准与服务流水线 | [训练器](https://github.com/Shalimov04/open-jev/blob/6126df4b8d9163af835048476a7eee0858b81dbb2/openjev/train.py)、任务配置、评测与多种子报告公开。每个产物有固定标签集，不是请求时任意更换候选的通用模型 |
| [Von · 7d0ff64](https://github.com/wfzyx/von/tree/7d0ff642a989510b2148c151e7109a3b9750fca5) | ModernBERT-large + option-marker 评分头；listwise CE + Brier；Choice/Noul/Score | [微调入口](https://github.com/wfzyx/von/blob/7d0ff642a989510b2148c151e7109a3b9750fca5/docs/finetune.md)、语料构建器、[权重与模型卡](https://huggingface.co/wfzyx/von)公开。适合研究 encoder 成本与校准；发布版本号变化可能只涉及推理引擎 |
| [daseinlabs/open-jev · 4627a04](https://github.com/daseinlabs/open-jev/tree/4627a04a075bf714f4b60b69de19fa2de0a8a431) | Gemma3-4B 冻结特征 + 可训练 cross-attention 头；另有候选 token 概率推理 | 提供特征抽取、头部训练和评测代码；头部训练路径使用 MLX，不能把其 PyTorch 推理支持当作 Linux 训练支持 |
| [Chinese-Jev · aed6193](https://github.com/gulucaptain/Chinese-Jev/tree/aed6193be5697488c8d0c6a77cc0c7e297329e61) | encoder typed-decision 微调；数据构建、训练、校准与评测 | 有[论文](https://arxiv.org/abs/2609.36965)与流水线代码；核查版 README 明确 Chinese-Jev 自有模型权重尚未附带。训练入口或论文结果不能替代已发布完整产物的证据 |
| [Verdict / Verdict 2.0 · bff2856](https://github.com/Heman10x-NGU/openJev-verdict-2.0/tree/bff28567cff463b833bf044f351a8b7945d53e07) | 前者为 ModernBERT/GLiClass；后者为 marker-pointer 分布头与独立 correctness 头 | 模型、训练、[损失代码](https://github.com/Heman10x-NGU/openJev-verdict-2.0/blob/bff28567cff463b833bf044f351a8b7945d53e07/verdict2/losses.py)及报告公开。核查目标包含 soft CE、Brier 和一致性项；README 的 RLCD 名称本身不能证明策略梯度。两仓库属于同一作者系列 |

## 选项 token 训练、蒸馏与多模态

这些实现通常复用原模型的选项 token logits，或以它初始化固定槽位头。没有长文本生成，不代表一定采用独立标量 ScoreHead。

| 项目与核查版本 | 训练方式 | 公开材料及限制 |
|---|---|---|
| [Mapika/decider · 4502408](https://github.com/Mapika/decider/tree/45024082b7d4bb667bf9140b7a7c073d3b483f6b) | Qwen3.5-2B-Base 选项 token CE；v11 加旧模型 KL replay；另有游戏 PPO | [监督训练](https://github.com/Mapika/decider/blob/45024082b7d4bb667bf9140b7a7c073d3b483f6b/decider/train.py)、[PPO](https://github.com/Mapika/decider/blob/45024082b7d4bb667bf9140b7a7c073d3b483f6b/decider/games/rl.py)、[权重](https://huggingface.co/Mapika/decider-2b)与报告公开；完整混合训练快照未核实 |
| [JevK5 · f26426d](https://github.com/allebee/jevk5/tree/f26426d16f59e8bbe1470e5b162cc89329e29b29) | Qwen3.5-4B/9B LoRA；teacher 合成/核验、公共数据 replay、选项分布 CE | [训练器](https://github.com/allebee/jevk5/blob/f26426d16f59e8bbe1470e5b162cc89329e29b29/training/lora.py)、[权重](https://huggingface.co/alibiserikbay/JevK5)与逐条评测公开；完整合成训练集发布未核实。默认目标不是 RL |
| [Bespoke Nimble · dcfdbd9](https://github.com/bespokelabsai/nimble/tree/dcfdbd9a64f0d869f658d7a72f1beaee32737773) | Qwen3.5-9B LoRA；最小事实改动形成答案翻转对照，候选硬标签 CE | 原始 2,676 train/324 eval、manifest、[训练器](https://github.com/bespokelabsai/nimble/blob/dcfdbd9a64f0d869f658d7a72f1beaee32737773/nimble/training/schema_train.py)与[权重](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B)公开。不是 Jev 蒸馏；标签由模型核验。原始配方对应 `original-2676`，不能混用 Hub latest |
| [Plumb · fe73067](https://github.com/crh225/plumb/tree/fe73067682729221bc61aac1db6c1e08f2f7d7f8) | JevK5/Qwen3.5-4B 派生训练；teacher 合成与核验、hard mining、replay、选项 CE | 管线、[权重](https://huggingface.co/crh225/plumb-4b)、[合成数据](https://huggingface.co/datasets/crh225/plumb-decisions)与结果公开。其数据集名为 `test` 的 131 条用于温度校准；公共 benchmark 汇总也影响过迭代，不属于完全未见最终评估 |
| [Reflex · 231f896](https://github.com/kshetrajna12/reflex/tree/231f896d818a62b94fec305ed553df1088486dcb) | Qwen3.5-4B；可选硬/软 CE、Brier 蒸馏与 anchor replay | 训练代码、旧 adapter 和负结果公开；当前推荐版本退回冻结基座与选项换序平均。不能将“存在微调代码”写成“微调后的发布模型更好” |
| [imajev · ccf586d](https://github.com/mohit67890/imajev/tree/ccf586d43d2a580319b6535c893668904d909eb9) | Qwen3.5 多模态，最新 4B 为 rank64 LoRA + 256-code 线性头；硬/软 CE，并有训练专用 rationale loss | [训练器](https://github.com/mohit67890/imajev/blob/ccf586d43d2a580319b6535c893668904d909eb9/scripts/train_decision_lora_torch.py)、adapter、读出权重与[发布规格](https://huggingface.co/mohit67890/imajev-4b/blob/f8d8234cebc6c99065c07731e59716dc0a6e27ab/RELEASE-SPEC.md)公开；完整训练快照未核实，合成视觉 benchmark 不能代表真实视觉任务 |

## 开放权重、推理服务与派生实现

| 项目 | 已核实的范围 | 使用时的区分 |
|---|---|---|
| [AnyJev · f82fe03](https://github.com/nokia-applied-research/AnyJev/tree/f82fe03bc0d01c9e340b4f8880b744e78ea6116d) | `Decider` 提供免训练的选项 logits 读出；新增 `Tacit` 自蒸馏权重和按置信度调用推理的路径 | 不能再仅归为冻结模型封装。[Tacit-1.7B 模型卡](https://huggingface.co/morriszjm/Tacit-1.7B/blob/1cb46fff0ffeefc4965455185074be5c62adccf7/README.md)描述自生成、自推理标签；本次未核实完整训练器与数据发布 |
| [SemIf-OpenJev · 23cf1f3](https://github.com/TheoLeeCJ/SemIf-OpenJev/tree/23cf1f39fc9534fe81437200959b6dfc7106e45a) | 冻结模型的答案槽位概率、多后端与温度校准 | 作为无微调基线；不据此宣称已完成任务模型后训练 |
| [NeoHorse-Jev · c494069](https://github.com/TokenRhythm/NeoHorse/tree/c494069accfc2d08839675961c1fdfcc54ba0b92/jev) | [4B backbone + pointer 权重](https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B)、推理与服务；runtime 声明源于 Kev | 尚未核实 Jev 专用完整训练入口和数据。NeoHorse 基座的训练论文不能替代这个任务头的训练证据 |
| [Hopper · b774218](https://github.com/hopit-ai/hopper/tree/b774218aa5193904509513a7ba95f12fa2607b61) | 原 Qwen3.5-4B 与新增 Gemma-4-12B 路径；服务、adapter、模型卡和评测 | 公共仓库未找到 trainer，[模型卡](https://huggingface.co/HopitAI/hopper/blob/4b605f25eb903c2cf4d121040ffb097240b3bc37/README.md)说明训练数据未分发。不同 adapter 的使用条件也需分别查看 |
| [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide/blob/5a7adf72a23b4d311abae6ce050d7f0012bb3416/README.md) | DeBERTa-v3-large，340M，动态标签单选/多选与多任务；公开权重和 [GLiNER2 工具包](https://github.com/fastino-ai/GLiNER2/tree/55656fbfa01d3d4a77485e1a1eeeaf682990ccdf) | 框架包含训练支持；本次未核实这份 Decide 权重的完整训练快照。可作为 encoder 任务模型参照 |
| [openjev/openjev 权重](https://huggingface.co/openjev/openjev/blob/ac97900fd034fdd7e7e536f3d4c21b836cae0750/README.md) | 27B 模型、选项字母概率、服务与模型卡；单次最多 52 个选项 | 与上述 GitHub 同名仓库分别核查。权重为 CC-BY-NC-4.0，服务代码为 Apache-2.0；属于有使用限制的开放权重，不能笼统称为完整开源训练工具包 |
| [logan-markewich/jeff · 34b32f9](https://github.com/logan-markewich/jeff/tree/34b32f99a727c47b679adde33f4702a001e02979) | 加载 GLiFormer 的 Jev 兼容服务 | 不是上文 firelex/jeff，也不是另一份独立训练模型 |
| [IluvatarLabs/kev · 45dc25e](https://github.com/IluvatarLabs/kev/tree/45dc25edee1e3e22bfa4beea4185fe2927e0bb86) | teacher → LoRA + pointer student 的转换、校准与导出代码；基于 Kev 的派生实现 | [NOTICE](https://github.com/IluvatarLabs/kev/blob/45dc25edee1e3e22bfa4beea4185fe2927e0bb86/NOTICE)声明源码来源；核查版仍称预训练下载待发布。不按 GitHub 非 fork 标记重复计算原创架构 |

## RL 实现的具体差别

- **JevForge**：在有限候选上优化任务效用、Brier 与参考策略 KL；代码包含精确期望和分组采样 REINFORCE 两种模式。[实现](https://github.com/zwliJay/jev-forge/blob/6cf0dfe3b811d623331c0cc8b1c00ff13614772e/jevforge/rlcd.py)。奖励来自已标注的有效动作，不能描述成浏览器在线探索；作者的 sampled 结果也报告了排序等级误差的退步。
- **Mapika/decider**：具有可执行游戏的 PPO 路线，环境交互与选项 CE 是不同训练阶段。
- **OpenJev-RLCD**：先采样生成推理，再对答案分布使用 proper-score reward 与 REINFORCE；其生成预算不等同于 Metis 的一次前向评分。
- **GLiClass**：提供分类动作采样、价值估计与 clipped policy objective 的 [RLTrainer](https://github.com/Knowledgator/GLiClass/blob/68132def761c2dccc20d1a71c04abdc5db35463f/gliclass/training.py)；存在训练器不等于已有目标任务的实测收益。

CE、Brier 或 KL 都可直接用于监督训练。采用 proper scoring rule、返回概率或在标题中使用 RLCD，不足以确认训练包含策略采样、奖励反馈或策略梯度。

## 对 Metis 的参考顺序

1. **结构与复现协议：JevForge、NanoJev。** 前者与独立候选 MLP 路径接近；后者具有同规模 0.6B 模型、公开程序目标和 episode 评测。模型版本、head 宽度、候选可见性和训练方式仍需对齐。
2. **工具包设计：Luce、S1LV3RJ1NX/openjev、GLiClass。** 参考任务定义、训练数据来源、专用适配、概率校准与产物接口；encoder 作为成本对照，不预设必须改变 Metis 的 decoder 主线。
3. **读出研究：Strands Decider、AgentJev、firelex/jeff。** 分别代表 pointer、候选集合残差交互和预训练读出初始化。应拆分实验，避免同时改变输入布局、可见性、参数预算和目标函数。
4. **数据与失败诊断：Nimble、Reflex。** 答案翻转对照检验是否利用证据；冻结基线、类别召回、校准和公开负结果检验训练是否真的改善任务表现。

这些参考没有改变 Metis 的质量结论：[第二轮 BoolQ 实验](../../cookbooks/boolq-readout-selection.md)尚未产生优于多数类基线的判别能力。应先确认任务训练能学到证据，再验证更复杂的读出或 RL；其他项目的自报分数不能替代本项目同协议对照。

## JevAny 的受控头部对照

JevAny 的 [pointer 算法](https://github.com/SimpleJev/JevAny/blob/2b9ed4c20f0e3d86db28bf7ddc1c11ac80c6dfe0/docs/ALGORITHM.md) 包含点积评分与非线性残差修正。其 [技术报告 §5.3](https://github.com/SimpleJev/JevAny/blob/2b9ed4c20f0e3d86db28bf7ddc1c11ac80c6dfe0/reports/JevAny_Tech_Report.pdf) 提供 250-step、904 个开发决策的头部对照：

| JevAny 评分头 | 参数量 | Accuracy ↑ | 校准后 NLL ↓ | Brier ↓ |
|---|---:|---:|---:|---:|
| MLP scorer | 2,688,001 | 82.41% | 0.4935 | 0.2596 |
| Residual pointer | 2,688,001 | 82.63% | 0.4586 | 0.2497 |

这是作者提供的短程单种子筛选，表中未报告差值区间。pointer 以决策表示与候选表示交互，残差输出层零初始化；报告未定义该对照 MLP 的具体输入，不能直接等同 Metis 的单 hidden-state MLP。此对照可支持实验设计，不能把 +0.22 pp 的开发集差值当作普适或已复现收益。

## 读出与可见性差异

Metis 的候选分支相互隔离，单个 raw score 只依赖公共 query/context 与自身候选。Kev/JevAny 默认隔离的是 question 分支；同一 question 内放置全部 options 和末尾 decision token，后面的 option 可读取前面的 option，decision token 可读取全部 options。[Kev 输入与 mask](https://github.com/jaredpalmer/kev/blob/5e42a7a03f28134853dd3ff77461457e921e5ec1/kev/model.py)、[JevAny 输入与 mask](https://github.com/SimpleJev/JevAny/blob/2b9ed4c20f0e3d86db28bf7ddc1c11ac80c6dfe0/jevany/model.py)。

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
