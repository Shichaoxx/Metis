# Metis 首轮 NFCorpus ScoreHead 训练协议（已执行，含首轮结果）

本轮目标是完成真实任务后训练、完整 dev 选型、sealed artifact 导出及调用闭环，并与原始 Qwen3-Reranker-0.6B 做质量对照。**Qwen3-0.6B-Base + 随机 MLP ScoreHead 是待训练方法，随机 head 不是强质量基线。** 不承诺超过原始 reranker 或达到 SOTA；Tree Mask 性能、dense retrieval 和其他任务不进入本轮。

已采用配置以 `recipes/nfcorpus/qwen3_base_score_head_06b.json` 为准。已核实纯 base checkpoint，并在 本地环境 完成真实权重的前向、梯度和重载预检。完整任务训练、dev 选型、最终留出评测和产物调用已在 本地环境 完成，状态见 [工作盘点](../docs/project/status.md)。

基座、数据和教学来源分别见 [来源致谢](../ACKNOWLEDGEMENTS.md)、[NFCorpus 数据条款](nfcorpus.md#数据与许可) 与 [参考资料](../docs/research/references.md)。本轮未使用 tree 训练。

## 冻结数据与监督

使用 `data/nfcorpus-bm25/manifest.json`，SHA-256 为 `31d952d490305e3973a2dfb5a28e2e57728dad8c5a74688e811df067e9831d24`。原生 train / dev / test 为 2,590 / 324 / 323 query，共享 3,633 篇文档；原始每 query 固定 BM25 top50，dev/test 不改候选。

| split | query 数 | 候选正标签总数 | 显式 0 标签 | 未写入标签的候选 | 注入正例 query |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 2,590 | 13,051 | 116,449 | 0 | 644 |
| dev | 324 | 1,415 | 0 | 14,785 | 0 |
| test | 323 | 1,527 | 0 | 14,623 | 0 |

上述数据来自本地 JSONL 的轻量统计。train 正例数中位数为 2，范围 1–46；1,403 个 query 只有 1–2 个候选正例。train qrels 全部等级为 1；dev/test 完整 qrels 含等级 1 和 2，候选监督 labels 二值化，正式 nDCG 仍读取完整原始 qrels。

- train 的 116,449 个 0 标签全部是 `weak_negative_ids`：未被 qrels 标为正例的文档，不是人工确认无关。即使样本写着 `unjudged_policy=ignore`，这些显式 0 仍参与训练。
- 644 / 2,590（24.86%）个 train query 在原始 top50 没有正例，准备器用一个 train qrels 正例替换末位候选。`injected_positive_ids` 保留来源；这是训练监督构造，不能算作检索命中。
- dev/test 从不注入正例，分别有 88 / 79 个 query 在候选集里没有正例，全部保留在评测均值中。unjudged 在标准 IR 评测贡献 0 gain，不等于人工负标签。
- dev 候选 labels 只有正例。旧式 Trainer dev BCE loss 只能反映正例分数，不能用于选型。本轮使用完整 324-query、完整 qrels 的 nDCG@10。

## 模型、精度和输入

| 项目 | 已采用协议 |
| --- | --- |
| 模型来源 | `Qwen/Qwen3-0.6B-Base`，revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`；加载本地 `data/models/Qwen3-0.6B-Base` |
| adapter family | `qwen3_score_head` |
| ScoreHead | Linear(hidden_size, 256) → GELU → Linear(256, 1)，有 bias；随机初始化由 seed 42 控制，初始化产物/参数证据记录到运行日志 |
| 读出 | 最后一个 relevance suffix token 的 hidden state，输出 raw relevance logit；不生成自然语言 |
| backbone 训练 | LoRA，q/k/v/o，r=8，alpha=16，dropout=0；ScoreHead 全量训练 |
| 参数与计算精度 | FP32 master weights，CUDA BF16 autocast；完整 dev 选型采用同一策略 |
| 输入布局 | pairs，SDPA，pair microbatch=4，单 pair **总输入上限 2,048 tokens** |
| Tree 设置 | recipe 保留 max_tree_tokens=4096，但本轮布局是 pairs，不构成 tree 加速实验 |

ScoreHead 使用独立的自然相关性模板：前缀 `<Task>: Estimate candidate relevance.`，后缀 `<Relevance>:`，以实际 sealed `input_spec.json` 保存的完整 instruction、模板和截断规则为准。它没有伪装成原始 Qwen reranker 的 yes/no chat prompt。

原始 reranker 使用其既有官方 yes/no 模板与读出。两者保持同一候选集和 **2,048 总 token 上限**，但模板和可能的 tokenization 不同，不能声称逐 token 输入相同或保留了完全相同的正文长度。报告各自 compiler、tokenizer、截断 metadata。这个比较评价两种完整方法，不能把总差值单独归因于 head、mask 或微调。

训练 artifact 以 FP32 参数保存；正式评测使用 `--dtype float32 --precision bf16`，复现 dev 选型的数值策略。直接将全部权重 cast 到 BF16 是另一种配置，不能悄悄替代。原始 reranker 可以导出 sealed full artifact，经同一 evaluator、同一 dtype / precision 测量。

## 训练抽样：fixed_per_run

只从已有 train 的 50-candidate 池抽样，池中含已记录的注入正例，不改原始 manifest。每 query 至多 8 个候选、至多 2 个正例，seed 42；不足 2 个正例的空位可用于显式 0 标签弱负例。没有显式标签的文档不自动转成负例。

当前实现对 `[seed, query_id, candidate_id]` 的 JSON 计算 SHA-256，分别对正例和弱负例按稳定 hash 排序后选取；选中后保留原始候选顺序与监督值。它从 BM25 top50 池中做固定的伪随机选择，没有额外优先挑最难负例。该选择在一次 run 内固定，**第二个 epoch 重复相同候选，不做按 epoch 重采样**；覆盖限制应如实报告。

按当前统计，每轮会使用 2,586 个 8-candidate query、3 个 6-candidate query、1 个 7-candidate query，共 20,713 候选；两轮共 41,426 次候选训练暴露，并非 41,426 个不同 query-document 对。少于 8 的四个 query 是弱负例不足，不填充重复样本。最终以运行生成的 `sampling.json` / `sampling.jsonl` 数字和 hash 核对。

抽样 lineage 保存所选 ID、标签、来源候选数与采样后内容 hash；原始 metadata 表达原候选池来源，不能把原 metadata 的列表长度当成抽样后的候选数。候选 ID、排序位置、注入标记及标签均不得进入模型文本。

## 有限预算与 dev 选型

| 项目 | 已采用配置 |
| --- | --- |
| seed / epochs | seed 42；最多 2 epochs，不做超参数搜索 |
| batch | query batch=1；gradient accumulation=8；约 648 optimizer updates，以 Trainer 实际步数为准 |
| 学习率 | LoRA 2e-5；ScoreHead 1e-3，独立参数组 |
| warmup / decay | warmup_ratio=0.05，weight_decay=0.01；其余以 resolved config / optimizer 实际日志为准 |
| 目标 | query 平均 pairwise logistic loss × 1.0 + BCE × 0.1，再对 query 求平均 |
| 保存 | save_steps=324；gradient checkpointing 开启 |
| 初始诊断 | 训练前做完整 324-query dev，标记 `initial_model_reference`、`eligible_for_best=false` |
| 选型 | 每个 epoch 结束做完整 dev nDCG@10，最多两个训练产物参与；严格更高才替换 best，完全相同保留较早产物 |

随机初始模型的完整 dev 是诊断，不能成为 best，也不替代已训练好的公开 reranker 作为质量对照。不要在看到 initial/dev 指标后为本轮新增超参数搜索。pairwise/BCE 都依赖弱负例假设，不能将 sigmoid 当作已校准相关概率。

实际 GPU、训练步数与耗时由运行日志记录。启动前已经完成真实权重的前向/梯度预检；任务训练从固定 seed 初始化开始，以实际参数更新记录和训练后重载验收。不能因预算不足只跑部分 dev/test 并声称完整。pair microbatch 不消除整个 query 的计算图保留成本，本轮真正减小单 query 内存的是候选抽样。

## 原模型对照与最终 test

先在相同 frozen BM25 top50 上测原始 reranker 的完整 dev；原模型使用 revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`，它不属于 base checkpoint。dev 报告主指标 nDCG@10（linear gain），附 MAP、全部 50 候选范围的 MRR、Recall@10、candidate recall 和截断统计。分母和 nDCG ideal ranking 取完整 qrels，失败与零召回 query 不能丢弃。

从两个已训练 checkpoint 中选出 dev 最佳 artifact 后，冻结模型、数据、代码、模板与选择理由，**该选定 ScoreHead 的完整 323-query test 只执行一次**。不在 test 比较两个 epoch，不因 test 落后重开配方。原始 reranker 在同一数值协议下作为冻结对照；ScoreHead 的 best 不意味着必须替换原始 reranker，退化也应完整报告。

历史原模型 MPS FP16 test 已完成：nDCG@10 = 0.35439461322471666，MAP = 0.1528469591896228，MRR = 0.5732320935447198，Recall@10 = 0.16593822162906427；候选 Recall@50 = 0.21171090624695924。13 / 16,150 个 pair 被截断。该历史分数已经被看过，不能把后续 test 描述成从未查看的盲测；能保证的是不再用 test 选配方或 checkpoint。

新的对照优先在相同 CUDA / FP32 weights / BF16 autocast 下运行两种 sealed artifact。训练与推理使用实际空闲、或经用户明确允许共享的 GPU；设备和环境分别记录。跨设备、精度或运行负载的结果不直接推断速度倍数。

## 评测脚本与证据

`examples/benchmarks/evaluate_artifact.py` 通过 Predictor 读取 sealed artifact，使用完整 manifest 和 qrels；记录 artifact/base 依赖/数据/源码 hash，逐 query 持久化，CUDA 前后同步，warmup 与模型加载单独记录，CUDA peak memory 明确范围。`--limit` 永远标为 pilot。`--resume` 只允许同身份与同协议恢复，失败不会生成成功子集汇总。

```bash
python examples/benchmarks/evaluate_artifact.py \
  --manifest data/nfcorpus-bm25/manifest.json --split validation \
  --artifact /absolute/path/to/sealed/export --output /new/evaluation/directory \
  --device cuda:0 --dtype float32 --precision bf16
```

最终 test 将 `--split validation` 替换为 `--split test`，使用已冻结的最佳 artifact 和新的目录。设备编号以该进程实际 CUDA 可见设备为准。

`compare_task_models.py` 比较同 manifest/split/qrels/query/candidate IDs、相同指标实现和总 token 上限的报告。它允许不同 compiler/readout，明确标记为不同完整方法的对照；不接受 pilot/full 混用。nDCG@10 paired bootstrap 固定 10,000 resamples、seed 42，报告绝对差值与 95% CI；CI 不能反馈进 test 选型，不证明 SOTA 或跨 seed 稳定性。

```bash
python examples/benchmarks/compare_task_models.py \
  --baseline /path/to/original-reranker-eval \
  --candidate /path/to/selected-score-head-eval \
  --output /new/comparison/directory
```

本轮真实执行结果与证据已补齐，见下一节。框架测试、真实权重预检和真实任务训练分别留档；不可互相替代。

## 首轮实际结果（2026-09-27）


| 完整324-query dev | nDCG@10 | 用途 |
|---|---:|---|
| 随机head初始 | 0.090762 | 初始化参照，不参选 |
| step324 | 0.242403 | 第一轮候选 |
| step648 | 0.263831 | 严格按dev选为best |

选择后冻结的best artifact seal SHA-256：`33a5e789723e3c63d09c9991816dc309831eb9741f06f309eb66adc86a477fca`。仅该best完成一次完整323-query test，16,150候选；未因test结果改配方或评测其他epoch。

| 完整test | 原始Qwen3-Reranker-0.6B | Base + ScoreHead + LoRA | 差值 |
|---|---:|---:|---:|
| nDCG@10 | 0.354783 | 0.290978 | -0.063805 |
| MAP | 0.152939 | 0.124735 | -0.028204 |
| MRR | 0.573162 | 0.476716 | -0.096446 |
| Recall@10 | 0.166522 | 0.149894 | -0.016628 |
| 候选Recall@50 | 0.211711 | 0.211711 | 0 |

nDCG@10差值的95% paired bootstrap CI为[-0.079595, -0.048809]，10,000 resamples，seed42。**工程闭环完成，但质量落后于本轮强基线。** 未实现“接近SOTA”的质量目标，不能将这一完整方法差异解释为ScoreHead或微调本身无效。下一轮应另定训练/验证方案；不能在本次test上追调分数。

Trainer耗时5849.8秒，含训练期间dev选型；训练模型test同步打分总计434.56秒，query平均1345.39ms，p95为1540.16ms，CUDA峰值allocated为2,882,312,192bytes（warmup后，含驻留模型）。训练模型截断3对，原始reranker截断13对；模板不同，总token上限一致不等于保留相同正文。性能只作本次记录，未作受控速度消融。

独立新进程在FP32 weights/BF16 autocast下重载best，对固定前三条dev的150候选复现分数，最大差0（预设atol1e-5、rtol0）。workflow使用该best返回候选ID并实际构建下游证据上下文；未调用通用LLM，不代表最终回答质量。


## 在 本地环境 执行训练与工具调用

以下保留复现入口；**本轮已完成，不要对已有run重复执行训练或test**。仅在用户安排新的独立实验时使用新输出目录执行。命令在项目根目录、具有 Torch / Transformers / PEFT 的独立环境中执行；正式模型计算使用 本地环境。公开权重和 NFCorpus 数据默认保存在被 Git 忽略的 data/，不随代码发布。已有的冻结数据继续复用，不重新生成候选。

```bash
# 先核实 GPU 资源；编号按实际可用设备选择。
nvidia-smi
# recipe 内固定了基座目录、版本、采样、训练预算和完整 dev 选型。
CUDA_VISIBLE_DEVICES=0 metis train recipes/nfcorpus/qwen3_base_score_head_06b.json
```

本轮已在 本地环境 准备固定版本的 `data/models/Qwen3-0.6B-Base` 与 `data/nfcorpus-bm25`。在另一台机器复现时，需下载同一公开模型 revision 并校验相同 manifest 身份；不能拿旧 `data/nfcorpus` 模板替代本轮 BM25 top50。

训练命令最后返回 run 与 selected artifact 路径。本配方返回 `exports/best`，同时保留 `exports/final`。`validation/initial.json` 只作初始化诊断，`selection.json` 记录 best 的 dev 指标与步数；`parameter_updates.json` 记录 head 和 LoRA 参数是否实际改变。可恢复 checkpoint 还依赖其引用的 `selection/best-step-N` 历史目录，应一同保留。

把下列路径替换成训练命令输出的 best artifact：

```bash
METIS_ARTIFACT=/absolute/path/to/run/exports/best
python examples/nfcorpus_workflow.py   --manifest data/nfcorpus-bm25/manifest.json --artifact "$METIS_ARTIFACT"   --output /new/workflow-trace.json --device cuda:0 --dtype float32 --precision bf16
```

该示例按固定顺序选第一条 validation query，去掉监督标签，将模型返回的候选 ID 交给下一步组装证据上下文，并保存 trace。它验证真实任务产物可供 Harness 调用，不宣称已验证通用 LLM 的最终答案质量。
