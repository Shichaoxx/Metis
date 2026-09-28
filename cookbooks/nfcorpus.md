# NFCorpus：原始 reranker 基线与保留的 yes/no 微调模板

**状态（2026-09-27）：** BM25 数据链路和原始 Qwen3-Reranker-0.6B 的完整 test 推理已完成。本页的 yes/no 微调与 embedding top100 路线未执行；当前真实后训练采用冻结 BM25 top50 上的 Qwen3 Base + 独立 ScoreHead，见 [本轮训练协议](score-head-plan.md)。本文件不是已完成的后训练成果；工具包的验收以 [CORE](../docs/project/core.md) 为准，进度见 [工作盘点](../docs/project/status.md)。

当前结果见 [项目状态](../docs/project/status.md)。不应据此声称 Metis 或所选 0.6B 基线达到当前 SOTA。Tree Mask 的教学来源与署名见 [来源说明](../ACKNOWLEDGEMENTS.md)。

这份 recipe 将 NFCorpus 的医学信息检索任务转成候选排序。本页保留的模型路径使用已经训练过的 `Qwen/Qwen3-Reranker-0.6B`，直接读取原有 LM head 的 `yes − no` logit，不随机初始化一个评分头。微调的目标是改善当前任务；训练成功、loss 下降、tree mask 可运行都不等于排序质量提高。

这里提供两条路线：**BM25 是无需下载模型的候选准备路线，其冻结 top50 已用于本轮 ScoreHead cookbook；Qwen3-Embedding-0.6B → top100 → Qwen3-Reranker-0.6B 是待实测的质量路线。** 后者不能凭模型名称称为 SOTA。相对强公开 reranker 的质量差距，需要相同候选、输入长度、完整 qrels 和模型版本下的测量来回答。

## 数据与许可

数据来自 [BEIR 官方 NFCorpus archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip)，原作者页面是 [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)，论文为 Boteva et al., *A Full-Text Learning to Rank Dataset for Medical Information Retrieval*, ECIR 2016。BEIR 是重排后的数据格式；原论文网页的全文档数量和 BEIR 版本不同。

已下载并校验的 archive 为 **2,448,432 bytes**，SHA-256 为：

```text
efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b
```

该版本包含 3,633 篇文档、3,237 个 query。原生 train/dev/test 分别有 **2,590 / 324 / 323** 个 query；在本工具内仅把 `dev` 改名为 `validation`，不重新随机切分。三份 qrels 分别有 110,575 / 11,385 / 12,334 条判断；train 的等级均为 1，dev/test 含等级 1、2。文档库跨 split 共享，query ID 不重叠。

[原作者 Terms of Use](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/#terms-of-use) 明确允许学术用途；其他 NutritionFacts.org 数据用途需要查看其服务条款并联系 Dr. Michael Greger。**工具包的代码许可不会替代数据条款。** 本仓库仅提供下载、准备脚本与元数据，不附带数据文本；预训练权重也不随仓库分发。

## 1. 运行轻量数据链路

在仓库根目录执行：

```bash
python -m pip install -e .
cometa prepare-nfcorpus --download --data-dir data/raw --output data/nfcorpus-bm25 --top-k 50
cometa dataset register nfcorpus-bm25 data/nfcorpus-bm25/manifest.json
cometa validate data/nfcorpus-bm25/manifest.json
cometa evaluate --manifest data/nfcorpus-bm25/manifest.json --split validation --baseline bm25 --output runs/bm25-dev
```

已有 archive 时可离线调用 `nfcorpus.fetch("data/raw", archive_path="/path/nfcorpus.zip")`，再运行不带 `--download` 的准备命令。下载器校验 archive 大小、SHA-256、成员白名单和每个原始文件的 SHA-256。原始数据留在 `data/raw/nfcorpus`，prepared 数据写到独立目录。

参考 BM25 的固定协议是：`title + "\n" + text`；小写后以 `[a-z0-9]+` 分词；无 stemming/停用词过滤；`k1=1.2, b=0.75`；每个 query term 只贡献一次；正 IDF `log(1+(N−df+0.5)/(df+0.5))`；按分数降序、文档 ID 升序打破平分；不足的有效词匹配用零分候选补齐到 K。**这是可复现的本地参考实现，不是 BEIR Elasticsearch/Pyserini BM25 数值复现。**


| split | query 数 | top50 平均候选 Recall | 无正例 query | 训练注入 query |
|---|---:|---:|---:|---:|
| train | 2,590 | 0.201613 | 644 | 644 |
| validation | 324 | 0.181959 | 88 | 0 |
| test | 323 | 0.211711 | 79 | 0 |

候选 Recall 分母包含全部 qrels 正例，分子是注入前的检索结果。这说明第一阶段本身有明显召回瓶颈：重排无法找回候选集之外的文档。不能把轻量路线上的 reranker 结果外推为全库检索能力。top50 准备输出约 276 MiB，因为 canonical JSONL 为每个 query 重复保存候选文本；代码仓库不保存生成数据。

## 2. 冻结更强的候选集

本节仍是未执行的 embedding top100/CUDA 路线，资源需求没有实测。后续运行优先使用已获授权的 本地环境 空闲 GPU；下面先运行 embedder，再准备训练数据。已有 MPS reranker 推理不构成本节已完成的证据。

```bash
python -m pip install -e '.[train,peft]'
python -m cometa.benchmarks.retrieval \
  --data-dir data/raw/nfcorpus \
  --output data/retrieval/nfcorpus-qwen3-06b-top100.jsonl \
  --model Qwen/Qwen3-Embedding-0.6B \
  --device cuda --dtype bfloat16 --batch-size 8 --max-length 8192 --top-k 100
cometa prepare-nfcorpus \
  --data-dir data/raw/nfcorpus --output data/nfcorpus --top-k 100 \
  --retrieval-run data/retrieval/nfcorpus-qwen3-06b-top100.jsonl
cometa validate data/nfcorpus/manifest.json
```

正式实验应把已核实的 HF commit 写入 embedding 命令的 `--revision`，并固定 reranker 的 revision。本次已测 reranker revision 是 `e61197ed45024b0ed8a2d74b80b4d909f1255473`，不能将其用于 embedding。训练 recipe 的 `revision` 仍为 null，且默认 manifest 是 `data/nfcorpus/manifest.json`；已测试的 BM25 数据则位于 `data/nfcorpus-bm25/manifest.json`，二者不可无说明地混用。代码另记录模型身份、依赖版本与输入哈希。

实现遵循 [Qwen 官方 Transformers 示例](https://github.com/QwenLM/Qwen3-Embedding)：query 是 `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query}`，文档不加 instruction；提取最后一个非 padding token、转 float32 后 L2 normalize，再对全部 3,633 篇文档做 exact cosine 检索。前向按 batch 执行，query 与 doc 向量各计算一次；相似度矩阵按 query batch 计算，不使用近似索引，不读取 qrels。

输出包含一个固定分数 JSONL 和同名 `.meta.json` sidecar。导入器检查分数文件 SHA-256、corpus/queries SHA-256、query 与 doc ID、候选数量和有限分数，并要求 `gold_used=false` 声明。它不会根据 dev/test gold 补候选。相同分数按 doc ID 打破平分。外部 retriever 也能导入相同格式；其来源、模型版本和“不使用 gold”的真实性由生产者负责，校验文件哈希不能证明上游实验设计。

```json
{"id":"PLAIN-1","scores":{"MED-1":0.71,"MED-2":0.62}}
```

示例只是格式；实际每个 native query 必须至少含 `min(top_k, corpus_size)` 个真实候选，且必须覆盖全部原生 split query。sidecar 至少包含 `method`、`run_sha256`、`source_file_sha256` 中的 corpus/queries hash 和 `gold_used:false`。由本脚本生成的 sidecar 还记录 prompt、pooling、精度、截断长度和模型版本。

## 3. 在同一候选集上先测原模型

```bash
cometa evaluate --manifest data/nfcorpus/manifest.json --split validation \
  --baseline retrieval --output runs/dense-dev
cometa evaluate --manifest data/nfcorpus/manifest.json --split validation \
  --model Qwen/Qwen3-Reranker-0.6B --layout pairs --backend eager \
  --device cuda --dtype bfloat16 --max-length 2048 --pair-batch-size 4 --output runs/pretrained-dev
```

先看完整 qrels 下的候选 recall，再看 nDCG@10、MRR、Recall@10。这里的 MRR 按全部提供的候选计算，未截断在第 10 名。若原模型效果已经足够，保留原模型也是合法结果。要比较更强 reranker，应另外冻结同一份候选并记录相同输入预算与 prompt；当前 recipe 不包含未实现的通用第三方 reranker adapter，也不以论文表格数字代替实测。

## 4. 小学习率微调，然后通过验证集门槛

主配置是 [qwen3_reranker_06b.json](../recipes/nfcorpus/qwen3_reranker_06b.json)：1 epoch，LoRA，学习率 `2e-6`，原 yes/no 读出，BCE + pairwise loss，pairs/eager 布局，单 pair 上限 2,048 tokens、pair microbatch 为 4、开启 gradient checkpointing。配置路径相对配置文件解析，默认读 `data/nfcorpus/manifest.json`。LoRA 与 checkpoint/export 已有 tiny CPU 验证；这不等于公开模型质量或 CUDA 资源验证。长度预算可能截断文档，原模型与微调模型保持同一预算，报告应检查截断 metadata。

```bash
cometa train recipes/nfcorpus/qwen3_reranker_06b.json
```

训练输出会给出实际 artifact 路径。将其代入以下命令：

```bash
ARTIFACT="/absolute/path/reported/by/train/export"
cometa evaluate --manifest data/nfcorpus/manifest.json --split validation \
  --artifact "$ARTIFACT" --device cuda --dtype bfloat16 --output runs/finetuned-dev
cometa compare --baseline runs/pretrained-dev --candidate runs/finetuned-dev \
  --metric ndcg@10 --max-drop 0
```

`compare` 检查 manifest、query ID、split、qrels、K 与候选协议一致；`--max-drop 0` 要求 nDCG@10 不低于原模型。结果相同也不证明改进：还需检查其他主指标、资源成本与多个 seed；决定采用微调模型应有预先设定的实际增益要求。这里的门槛是工程回退规则，没有实现显著性检验。默认保存最终 artifact，Trainer 的 dev loss 不自动决定“最佳模型”；不同 checkpoint 或超参数只能通过 validation 比较选择。

**验证不通过就保留原模型。** 选定方案后只在 test 上报告一次最终结果；不要看 test 分数后再次调超参数。原模型与最终模型的 test 对比可以保留作为最后报告，但不能把它用作选择器。

```bash
cometa evaluate --manifest data/nfcorpus/manifest.json --split test \
  --artifact "$ARTIFACT" --device cuda --dtype bfloat16 --output runs/selected-test
```

如果选择的是原模型，用 `--model Qwen/Qwen3-Reranker-0.6B --layout pairs --backend eager --max-length 2048 --pair-batch-size 4` 替换 `--artifact`，并使用与 dev 相同的 revision。

## 训练监督与评测边界

- **训练候选：** 默认沿用固定检索 top K；仅当其中完全没有正例时，用一个 qrels 正例替换最后候选。正例按 grade 降序、ID 升序选取，候选总数仍为 K。`injected_positive_ids` 逐样本记录操作，原始检索分数另存，统计使用注入前结果。
- **弱负例：** train 中未在 qrels 出现的候选显式标成 0，并保存 `weak_negative_ids`。它们是训练假设，不是人工确认无关。原始正例 grade 二值化供 BCE 使用；完整原始 graded qrels 单独保留。可研究噪声鲁棒目标，但不能把本协议写成完全标注数据。
- **dev/test：** 不注入任何正例。未判断候选不写入 supervision labels，`unjudged_policy=ignore`；没有召回正例的 query 也保留。训练期间仅对可计算的候选标注算 dev loss，单独日志记录跳过数量；正式 ranking evaluation 覆盖全部 query。
- **完整分母：** Recall/MAP 等分母用完整 qrels，nDCG 的 ideal ranking 也来自完整 qrels，未检索到的正例仍然算缺失。采用标准 IR 的 unjudged→zero gain 评测约定，不表示数据已经证明它是负例。graded nDCG 使用 TREC 风格 linear gain；报告里记录这项选择。
- **协议冻结：** `manifest.json` 保存原始与派生文件 SHA-256、候选生成参数、split 对应关系、许可和监督来源。冻结后不能只替换 JSONL 而沿用旧 manifest。训练/评测/部署共用 compiler；数据 target 与 metadata 不拼入模型输入。

## 资源与后续消融

pairs 按 query 的候选分成较小 microbatch，训练配置也开启 gradient checkpointing；跨候选 loss 仍需保留相应计算图，不能把 microbatch 视为“显存完全不随候选数量增长”。梯度累积不能降低单个 query 的显存。先在目标 GPU 做少量 query 的资源 smoke，再跑完整实验；不要通过悄悄缩短 test 候选来掩盖资源问题。若需要减少训练负例，应单独生成带明确协议和哈希的训练配置，保留原 dev/test top100。

tree 方案适合研究共享 query 计算、候选隔离和一致 position IDs。先验证 pairs/tree 同输入的分数与梯度近似等价、候选置换稳定，再在同硬件测延迟和显存。当前 dense mask 实现不能仅凭“理论稀疏”声称已经获得稀疏计算加速；输入长度与后端也可能影响性能。tree 的工程收益与微调的质量收益需要分别报告。

已验证数据契约、tiny 训练与 fake embedding 链路；也已完成原始 Qwen3-Reranker-0.6B 在冻结 BM25 top50 上的 323-query test 和本机 MPS 小样本布局计时。**真实 embedding top100、真实 0.6B 任务微调、CUDA 性能和每布局峰值内存仍未验证。** 预训练推理和 tiny 参数更新不能合并表述成真实后训练 cookbook 已完成；下一次实施按 CORE 的验收收尾，不自动扩展其他实验。
