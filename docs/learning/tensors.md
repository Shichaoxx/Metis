# Metis：从 tensor 看懂一个可训练的决策模块

本文提供张量语义与现有参考实现说明；产品要求见 [CORE](../project/core.md)，当前验证范围见 [工作盘点](../project/status.md)。本篇讲解 Qwen3 yes/no 读出；新接入的 Qwen base + 独立 ScoreHead 见 [ScoreHead 张量说明](../architecture/score-head.md)，两者均有模型适配路径。PDF 保留初版快照，本文的状态说明较新。

**来源与致谢：** 本教程的 Tree Mask / Jev 讨论与张量化表达，参考了 **Bilibili「五道口纳什」的视频**（[作者主页](https://space.bilibili.com/59807853)）和其 [GitHub 教学资料](https://github.com/wdkns/modern_genai_bilibili)，特别是 `slides/jev_architecture-v2.pdf`。例子与图形由本项目重新构造、独立绘制；教学思路受到上述资料启发，已有算法不归为本项目首创。参见 [R0](../research/references.md#r0-五道口纳什的视频与-github-教学资料) 与 [版权边界](../../ACKNOWLEDGEMENTS.md)。

本文配套 `tensor-guide.pdf`。项目现名为 Metis，PDF 保留旧代号 Cometa。研究的问题是：怎样把预训练模型用于固定判断任务，并让输入编译、训练、评测和导出遵守同一份契约。它不是 Jev 的复现，也不声称发现了新的 attention 算法。

本篇的读出示例使用 **Qwen/Qwen3-Reranker-0.6B 的已有 yes/no head**；当前真实后训练采用 **Qwen3-0.6B-Base + 独立 ScoreHead**，见 [ScoreHead 说明](../architecture/score-head.md) 与 [训练协议](../../cookbooks/score-head-plan.md)。公开数据示例使用 **BEIR NFCorpus**。本文所有短 token、矩阵和分数均为合成教学例子；它们不是模型预测、性能结果或业务数据。Jev 架构研究材料只提供问题背景，不能据此确认 Jev 的内部结构。

## 1. 先看输出：一次 forward，得到每个候选的分数

对于一个 query 和 K 个候选，目标是：

```text
query + candidates
        ↓ compiler
token IDs + mask + positions + readout mapping
        ↓ pretrained decoder
hidden states at each candidate's decision position
        ↓ existing yes/no vocabulary projection
scores: [K] → rank / threshold / downstream workflow
```

这仍然是语言模型的权重和词表读出，但请求不必调用 `generate()`，也不需要采样自然语言解释。每个执行块做一次前向；超出物理长度预算时分块计算，详见第 7 节。执行环境、候选召回、工具调用和业务约束仍由 Workflow / Harness 负责。

统一样本的最小推理输入：

```python
sample = {
    "schema_version": "1.0",
    "id": "toy-q1",
    "task_id": "rerank",
    "input": {"query": "Which passage explains photosynthesis?", "context": ""},
    "candidates": [
        {"id": "toy-d1", "text": "Plants convert light into chemical energy."},
        {"id": "toy-d2", "text": "A train timetable lists departure times."},
    ],
}
```

候选 ID 是业务身份；候选在本次 tensor 中的下标只是临时位置。换序、padding、截断后，都必须能把输出还原到正确 ID。

## 2. yes/no 读出：不能把两个 softmax 混为一谈

Qwen 官方示例使用固定 system prompt、`<Instruct>/<Query>/<Document>` 内容和 assistant suffix，读取 suffix 最后一个输入 token 位置的下一 token logits。[R1]

设末层隐藏状态为 `H ∈ R[B, N, D]`，第 i 个候选的决策位置为 `r_i`：

\[
h_i = H[b_i,r_i,:],\qquad z_{i,y}=W_{yes}h_i,\quad z_{i,n}=W_{no}h_i.
\]

如果模型输出层含 bias，需要把对应 bias 计入；Qwen 的具体结构以加载的模型配置为准。保留原词表头的两行，与先计算完整词表 logits 再取这两行，在相同权重下等价；能否避免构造完整词表 tensor 由实现决定。

当前 Metis yes/no 适配器的原始排序分数定义为：

\[
s_i=z_{i,y}-z_{i,n},\qquad p_i=\frac{e^{z_{i,y}}}{e^{z_{i,y}}+e^{z_{i,n}}}=\sigma(s_i).
\]

例如两个候选的 `(z_no,z_yes)` 分别为 `(0,2)`、`(1,0)`，则 `s=[2,-1]`，`sigmoid(s)≈[0.881,0.269]`。这些数仅用于演示公式。

- **yes/no softmax**：每个候选内部的二分类归一化；多个候选可以同时相关，其概率不要求加起来等于 1。
- **candidate softmax**：在 K 个候选之间归一化，只适用于“恰好选一个”等明确的任务语义。
- **排序**：直接按 `s_i` 排序。sigmoid 单调，使用 `p_i` 不改变排序，但不能因此宣称分数已经校准。

批处理时，若有效候选总数为 C，gather 后的隐藏状态为 `[C,D]`，两个 logits 为 `[C,2]`，原始分数为 `[C]`。恢复样本分组后，分数也可以暂存为 `[B_sample,K_max]`，另带同形状的 `candidate_mask`；执行 batch 与业务样本 batch 不必一一对应。无效候选不进入 loss 和指标。不能盲目使用 `logits[:, -1, :]`：那只在有效决策位置确实都在最后一列时成立；通用 compiler 应显式保存 readout positions。

## 3. Tree mask：共享前缀与分支因果关系

考虑共享前缀 `P=[p0,p1,p2]`，三个候选分支各含两个 token：`A=[a0,a1]`、`B=[b0,b1]`、`C=[c0,c1]`。每个分支的最后一个 token 是此玩具例子的 readout。真实分支包括 document 内容和完整 assistant suffix。

```text
physical index: 0 1 2 3 4 5 6 7 8
token:          p0 p1 p2 a0 a1 b0 b1 c0 c1
group:          0 0 0 1 1 2 2 3 3
readout:                ^     ^     ^
```

mask 的行是 query token，列是 key/value token。允许关系为：

\[
M_{ij}=\mathbf{1}[j\le i]\,\mathbf{1}[g_j=0\;\lor\;g_j=g_i].
\]

对应 0/1 矩阵：

```text
      p0 p1 p2 a0 a1 b0 b1 c0 c1
p0     1  0  0  0  0  0  0  0  0
p1     1  1  0  0  0  0  0  0  0
p2     1  1  1  0  0  0  0  0  0
a0     1  1  1  1  0  0  0  0  0
a1     1  1  1  1  1  0  0  0  0
b0     1  1  1  0  0  1  0  0  0
b1     1  1  1  0  0  1  1  0  0
c0     1  1  1  0  0  0  0  1  0
c1     1  1  1  0  0  0  0  1  1
```

三种语义必须分清：

| 布局 | 分支能看共享前缀 | 分支能看其他分支 | 适用性 |
|---|---:|---:|---|
| 普通 causal list | 是 | 能看前面的分支 | 存在随排列改变的上下文 |
| 单纯 block diagonal | 否 | 否 | 丢失 query，不能直接替代本任务 |
| shared-prefix tree | 是 | 否 | 与独立 query-document 评分对齐 |

这里解决的是分支之间的信息可见性差异，不是保证任意任务都应该取消候选交互。如果业务需要候选之间的联合约束，应显式增加集合层或后处理。

## 4. Position IDs：物理地址不是逻辑距离

仅改变 mask 还不够。若 B 分支沿用拼接下标，其 token 到 query 的 RoPE 距离会包含前面 A 分支的长度。候选换序后，这个距离就变了。

```text
physical index: 0 1 2 | 3 4 | 5 6 | 7 8
correct pos:    0 1 2 | 3 4 | 3 4 | 3 4
wrong pos:      0 1 2 | 3 4 | 5 6 | 7 8
```

分支不是从 0 重新开始；它从共享前缀长度 S 开始。对分支 b 的第 t 个 token，`position_id=S+t`。这个位置与独立输入 `[P; branch_b]` 一致。

RoPE 使注意力内积依赖相对位置：[R2]

\[
(R_{p_i}q_i)^\top(R_{p_j}k_j)=q_i^\top R_{p_j-p_i}k_j.
\]

例如 `b1` 关注 `p2`，正确距离为 `2-4=-2`，错误拼接位置给出 `2-6=-4`。mask 虽然完全一样，attention 数值仍可能不同。

**等价性还要求 token IDs 一致。** BPE/tokenizer 在字符串边界附近可能改变切分；“同一段字符串”不自动等于“同一段 token 前缀”。应检查 pairs 与 tree 两条路径中每个分支对应的完整 token 序列，而不是靠肉眼比较模板。

## 5. 分块 attention 的 softmax 需要合并归一化常数

Tree 描述可见性；计算实现可以把可见 keys 分成共享前缀 P 和本分支 B。但不能分别 softmax 后把两个输出直接相加。

设某个 query 行在各分区的 logits 为 `a_P`、`a_B`，定义：

\[
\ell_P=\operatorname{logsumexp}(a_P),\quad
A_P=\operatorname{softmax}(a_P)V_P
\]

\[
\ell_B=\operatorname{logsumexp}(a_B),\quad
A_B=\operatorname{softmax}(a_B)V_B.
\]

合并结果为：

\[
A=\frac{e^{\ell_P}A_P+e^{\ell_B}A_B}{e^{\ell_P}+e^{\ell_B}}.
\]

实际计算使用 `m=max(ell_P,ell_B)`，以 `exp(ell_P-m)`、`exp(ell_B-m)` 避免溢出。

一个标量 value 通道的合成例子：

| 分区 | 未归一化权重 | values | 总权重 | 局部输出 |
|---|---|---|---:|---:|
| prefix | `[2,1]` | `[10,4]` | 3 | 8.0 |
| branch | `[3,2]` | `[2,6]` | 5 | 3.6 |

正确结果是 `(3×8 + 5×3.6)/(3+5)=5.25`，不是 `8+3.6=11.6`。向量 value 的每个维度遵守同一公式。本文推导说明 kernel 需要保留的信息，不表示当前原型已经实现了这种优化 kernel。[R3][R4]

## 6. 训练：共享前缀必须收到所有分支的梯度

对于有判断标签的候选集合 J，一条 query 的 BCE 可以写为：

\[
L_{BCE}=\frac1{|J|}\sum_{i\in J}\operatorname{BCEWithLogits}(s_i,y_i).
\]

对于明确正负或等级顺序的候选对 `y_i>y_j`，pairwise logistic loss 可以写为：

\[
L_{pair}=\operatorname{mean}_{y_i>y_j}\operatorname{softplus}(-(s_i-s_j)).
\]

最后先在每条 query 内归一化，再在有效 query 间平均。否则候选多的 query 会无意中获得更高权重。没有有效 pair 时可以跳过 pair 项并记录；不能因此把仍有 BCE 监督的整条样本删掉。缺少判断标签不自动等于负例。多等级 relevance 用于 pairwise 顺序；当前训练实现的 BCE 使用 `relevance > 0` 二值化，并把该规则写入导出配置。

共享 prefix 在计算图中只有一份，各分支梯度在这里求和：

\[
\frac{\partial L}{\partial H_P}=\sum_b\frac{\partial L_b}{\partial H_P}.
\]

推理时可复用 KV cache；训练时若 detach 共享 prefix 的 KV，就切断了这部分梯度，不再等价于完整训练。

正确性测试建议依次检查：

1. pairs 与 tree 每个候选的 token、position、readout 语义相同。
2. 关闭 dropout 等随机操作，相同权重下比较 raw scores。
3. 使用相同 labels 和 loss reduction，比较可训练参数梯度和一次 optimizer 更新。
4. 换序后按 candidate ID 还原，比较分数。
5. full / LoRA 导出重载后再比较分数。

`model.eval()` 不会禁止 autograd；可以在禁用 dropout 的同时执行梯度测试。不同 dtype/kernel 使用明确容差，不能把浮点误差当成语义差异。随机 dropout 下，重复计算前缀与共享一次前缀的单次噪声可能不同，不能无条件要求逐位相等。

默认训练引擎是 Transformers Trainer 的任务薄封装，不是 token SFTTrainer。自定义字段、label mask 和变长候选输出需显式适配；输入 metadata 不应误传 backbone，也不能被默认字段裁剪悄悄丢掉。[R5]

## 7. 计算量：逻辑稀疏不等于已经加速

令共享前缀长 S，候选数 K，第 b 个分支长 T_b：

```text
pairs 的处理 token 数 = K*S + sum(T_b)
tree 的处理 token 数  = S + sum(T_b)
节省的重复前缀 token = (K-1)*S
```

causal 有效 attention 边数：

\[
E_{tree}=\frac{S(S+1)}2+\sum_b\left(T_bS+\frac{T_b(T_b+1)}2\right).
\]

相对独立 pairs，减少的是重复 prefix-to-prefix 项 `(K-1)S(S+1)/2`；branch-to-prefix 的交互依然需要计算。MLP 和 Q/K/V 投影也可能因减少重复 prefix 而受益，但总加速比必须实测。

在 `S=3,K=3,T_b=2` 的玩具例子中，pairs 处理 15 个 token，tree 处理 9 个；有效边从 45 降为 33。然而，如果 reference 实现构造完整 dense score matrix，tree 仍可能分配 `9×9=81` 个格子，而三个独立 pair 是 `3×5×5=75` 个格子。因此 tree reference 的职责首先是证明正确性。

| 层次 | 决定什么 | 不能直接推断什么 |
|---|---|---|
| layout / mask | 哪个 token 能看谁 | GPU 是否跳过屏蔽区域 |
| positions | RoPE 的逻辑距离 | 自动保持任意模板的等价性 |
| eager / SDPA | attention 的执行方式 | 任意 mask 都走最快内核 |
| FlexAttention / 专用 kernel | 利用受支持的稀疏/共享结构 | 当前模型已适配、端到端必然更快 |
| 训练或推理图 | 是否回传梯度、是否保存激活 | 推理缓存能直接用于训练 |

PyTorch SDPA 的布尔 mask 中 True 表示允许参与；其他接口可能采用相反约定，转换时必须核对。[R3] 标准 FA2 的常规 causal API 不接受任意树形 mask。原型的支持矩阵、设备约束和错误提示以代码及 README 为准；未实现的 backend 应明确拒绝，不能静默换成别的算法。

当前原型实际提供 eager 与 SDPA；`flex` 明确抛出未实现错误。Tree 使用 dense 4D reference mask，按 `max_tree_tokens` 的物理 token 预算分块；`max_length` 限制每个独立 pair 的内容长度。每个块共享一份 prefix，但不同块会重新计算 prefix，因此不能把“每块共享”宣传成“整个请求永远只算一次 query”。这一实现先验证语义，不承诺稀疏加速。

性能报告至少固定候选数、输入长度、batch、dtype、硬件、tokenize 是否计时、warmup 与 GPU 同步方式，并同时报告质量、峰值显存和延迟。没有测量就不填“加速倍数”。

## 8. NFCorpus 操作路线：先建立可信基线

完整命令以 `cookbooks/nfcorpus.md` 与 `python -m cometa --help` 为准。NFCorpus 的 BEIR 版本提供 native train/dev/test；dev 在本项目 manifest 中映射为 validation。这里有两条路线：**冻结 BM25 top50 已用于本轮 ScoreHead 后训练与原始 reranker 对照；Qwen3-Embedding-0.6B exact cosine top100 → Qwen3-Reranker-0.6B 仍是待实测路线。** 两条路线的候选都要冻结，测试时不能把未召回的 gold positive 补进去。[R6]

建议依次跑：

1. 准备公开数据、固定 native splits，记录 hash、candidate top-k 与缺失标签策略。
2. 记录第一阶段候选召回，保留完整 qrels 作为指标分母；不要把 BM25 smoke 的低召回外推成模型能力。
3. 在同一候选集评测原始 Qwen3-Reranker；这是可用模型基线。
4. 短程训练与 export/reload 验证，再评测 fine-tuned 模型。
5. 权重固定，比较 pairs 与 tree 的正确性和性能；不要把换权重与换 backend 的收益混在一起。

仓库根目录下的 CLI 路线如下。首个命令下载小型公开数据；模型评测与训练会另外加载较大的预训练权重，不属于无需下载的 smoke test。输出目录要求不存在，重跑时使用新的 run 名称。

```bash
python -m cometa prepare-nfcorpus --download \
  --data-dir data/raw --output data/nfcorpus-bm25 --top-k 50
python -m cometa validate data/nfcorpus-bm25/manifest.json
python -m cometa evaluate --manifest data/nfcorpus-bm25/manifest.json \
  --split validation --baseline bm25 --output runs/bm25-dev
```


```bash
python -m cometa.benchmarks.retrieval \
  --data-dir data/raw/nfcorpus \
  --output data/retrieval/nfcorpus-qwen3-06b-top100.jsonl \
  --model Qwen/Qwen3-Embedding-0.6B \
  --device cuda --dtype bfloat16 --batch-size 8 --max-length 8192 --top-k 100
python -m cometa prepare-nfcorpus \
  --data-dir data/raw/nfcorpus --output data/nfcorpus --top-k 100 \
  --retrieval-run data/retrieval/nfcorpus-qwen3-06b-top100.jsonl
python -m cometa evaluate --manifest data/nfcorpus/manifest.json \
  --split validation --model Qwen/Qwen3-Reranker-0.6B \
  --layout pairs --backend eager --device cuda --dtype bfloat16 \
  --max-length 2048 --pair-batch-size 4 \
  --output runs/pretrained-dev
python -m cometa train recipes/nfcorpus/qwen3_reranker_06b.json
```

embedding 脚本按官方 query instruction、最后一个非 padding token pooling、L2 normalization 生成向量，对全库做 exact cosine；它不读取 qrels。候选分数与模型/输入 metadata 一起冻结。其当前验证只覆盖 fake embedding 的接口链路，不代表真实模型质量。

训练返回 artifact 路径后，用 `evaluate --artifact 实际路径` 替换上面的 `--model ...` 评测同一 validation 集；可用 `cometa compare --baseline runs/pretrained-dev --candidate runs/finetuned-dev --metric ndcg@10 --max-drop 0` 检查无退化门槛。验证不通过就保留原模型。模型选择只使用 validation，最终配置确定后再运行 test。top100 的单 query 训练仍可能占用大量显存，梯度累积不能降低这一条 query 的激活峰值。

本 recipe 只在 train 的固定 top-k 内没有正例时，用最高 relevance grade（再按 ID 打破平局）的正例替换最后一个候选，保持候选数不超过 top-k，并记录这个变换。dev/test 不注入正例，也不删除没有已判断候选的 query。train 中未出现在 qrels 的候选显式作为弱负例；dev/test 中省略其标签并标记 ignore。弱负例不等于人工标注负例。

NFCorpus 原作者的条款允许学术用途免费使用，其他用途需取得原作者许可；不能将它描述成通用商用开源数据。见 [原站 terms of use](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/#terms-of-use)。

最小评分调用（会在首次运行时按环境配置加载模型权重）：

```python
from cometa.model import QwenReranker

model = QwenReranker(
    "Qwen/Qwen3-Reranker-0.6B",
    layout="pairs",
    backend="eager",
    device="cpu",
    dtype="float32",
    max_length=4096,
)
scores = model.score([sample])[0]
ranked = sorted(zip(sample["candidates"], scores), key=lambda x: x[1], reverse=True)
print([(candidate["id"], score) for candidate, score in ranked])
```


**项目价值如何表达：**实现一个能复用预训练判别能力、保持候选输入语义、支持可验证训练和导出的工具包；列排序是启发来源，文档重排是公开示例。可以把它接到 Agent 的固定判断节点，但这不等于复现了 Jev，也不需要把整个 Agent 都改造成一个模型。

参考资料与可复查链接见 `references.md`。PDF 中的图为本项目原创矢量图；构建脚本为 `build_tensor_guide.py`。
