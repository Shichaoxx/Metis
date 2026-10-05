# NFCorpus 数据准备

NFCorpus 是 Metis 的首个公开任务：给定查询，对固定的文档候选打分排序。真实训练、选优、结果和调用见 [ScoreHead cookbook](score-head-plan.md)。

## 数据与许可

数据使用 [BEIR NFCorpus archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip)，来源和条款见 [NFCorpus 原站](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)。原作者允许学术用途；其他用途需遵守原站条款。仓库提供准备代码，不附带数据文本。

该版本有 3,633 篇文档，原生 train/dev/test 分别为 2,590 / 324 / 323 个查询。工具内将 dev 映射为 validation，保留原始划分与完整 qrels。压缩包大小为 2,448,432 bytes，SHA-256 为：

```text
efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b
```

## 准备 BM25 候选

在项目根目录执行，输出目录使用尚不存在的新目录：

```bash
metis prepare-nfcorpus --download --data-dir data/raw --output data/nfcorpus-bm25 --top-k 50
metis dataset register nfcorpus-bm25 data/nfcorpus-bm25/manifest.json
metis validate data/nfcorpus-bm25/manifest.json
```

已有数据时去掉 `--download`。准备器检查 archive 大小、成员和文件哈希。新准备的数据保存自己的 manifest；首轮实验继续使用原有冻结数据，避免覆盖历史身份。

BM25 使用 `title + newline + text`，小写后按 `[a-z0-9]+` 分词，`k1=1.2`、`b=0.75`，不做 stemming 或停用词过滤。每个查询词贡献一次，IDF 为 `log(1 + (N - df + 0.5) / (df + 0.5))`。候选按分数降序、文档 ID 升序排列，不足 top50 时补零分文档。这是本地参考实现。

## 候选与标签

- train 在 top50 完全没有正例时，用一个 qrels 正例替换末位候选；首轮共有 644 个查询发生注入，样本记录 `injected_positive_ids`。
- train 中未判断的候选作为弱负例，记录 `weak_negative_ids`；这些 0 标签是训练假设。
- dev/test 不注入正例。没有召回正例的 88 / 79 个查询仍保留，评价使用完整 qrels；nDCG 使用 linear gain。

首轮固定候选 Recall@50 为 dev 0.181959、test 0.211711。候选外的正例仍计入完整评价分母，重排模型只能排列已提供的文档。

manifest 记录 split、候选生成参数和文件哈希，训练、评测与调用使用同一输入编译规则。冻结后的候选、qrels 和 recipe 保持原值。
