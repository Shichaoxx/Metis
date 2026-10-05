# Qwen Base + ScoreHead：从隐藏状态到候选分数

**模型读完 query 和一个候选，直接给出一个可训练的相关性分数。** 这个分数可以用于列排序、候选选择或下游 workflow；整个路径不调用 `generate()`，不要求模型输出自然语言，也不依赖 yes/no 词表读出。

本文对应 Metis 的 `qwen3_score_head` 适配器。项目目标见[核心设计](../project/core.md)，实现见[模型](../../src/metis/model.py)、[输入编译](../../src/metis/compiler.py)和[模型工厂](../../src/metis/model_registry.py)。Tree Mask 与张量化讲解受到五道口纳什教学资料启发，来源见[致谢](../../ACKNOWLEDGEMENTS.md)。

以下两图展开首轮排序配方的架构与张量变化。`b` 是候选微批大小，`N` 是单个 query 的全部候选数；图中的色块示意轴结构，不代表实际维数或实验数值。两图统一配色：青色为离散输入或二值支持，紫色为隐藏激活，橙色为 Head 参数，红色为 raw logits 或损失，灰色为位置、标签或字符串列表。训练图的差分矩阵另用红、紫、白表示正、负、零；红色虚线表示反向梯度。

## 训练张量流

输入经 Base 和独立 ScoreHead 得到候选分数，汇总当前 query 的微批后计算监督损失；这些微批的计算图保留到反向传播。反向传播更新 q/k/v/o 的 LoRA 与 Head；Base 参数冻结，激活仍参与求导。

![Metis 训练架构、张量形状与反向梯度](figures/metis-tensor-training.png)

[矢量 PDF](figures/metis-tensor-training.pdf) · [可编辑 LaTeX](figures/metis-tensor-training.tex)

## 推理张量流

同一架构在 `eval()` 和 `no_grad()` 下运行。所有候选的 raw logits 按输入顺序拼接，再转换成 Python 分数，排序并返回字符串 ID；Workflow 按 ID 查回候选文本。

![Metis 推理架构、张量形状与候选选择](figures/metis-tensor-inference.png)

[矢量 PDF](figures/metis-tensor-inference.pdf) · [可编辑 LaTeX](figures/metis-tensor-inference.tex)

## 一次前向，三个张量

当前[训练配方](../../recipes/nfcorpus/qwen3_base_score_head_06b.json)使用 **Qwen3-0.6B-Base + 256 维 MLP**。基座通过 `AutoModel` 加载，不附带 LM head；MLP 是新建、需要训练的任务头。**本轮训练使用 pairs/SDPA；tree 是单独的正确性预检，没有参与这轮训练质量或加速对照。**

`B` 是一次 backbone 前向的物理 batch 大小，`L` 是该执行块的 token 长度，`D` 来自模型配置，`C` 是该块的有效候选数。它们不等于训练时的业务 query batch：pairs 可以一次放多个独立序列；tree 通常是一个物理序列包含多个候选分支。分块结果最终按输入候选顺序拼回。

对第 (i) 个候选，读出位置为 ((b_i,r_i))：

\[
h_i=H[b_i,r_i,:],\qquad
s_i=W_2\operatorname{GELU}(W_1h_i+b_1)+b_2.
\]

其中 (W_1\in\mathbb{R}^{256\times D})、(W_2\in\mathbb{R}^{1\times256})。输出 `raw_relevance_logit` 是未约束的标量；排序可直接使用它，sigmoid 后的值也不自动成为经过校准的概率。`256` 是本配方的设置；`head_hidden_size` 可配置，未指定时为 `max(1, D // 4)`。

## 读出放在候选之后

[ScoreHeadCompiler](../../src/metis/compiler.py)使用普通 relevance 模板。下例仅展示模板结构：

```text
<Task>: Estimate candidate relevance.
<Instruct>: Score how relevant the candidate is to the query and its context
<Query>: 用户问题
<Document>: 候选描述
<Relevance>:
             ↑  最后一个输入 token 的 hidden state → ScoreHead
```

Context 存在时进入 instruction 部分。`<Relevance>:` 是正常分词的模板文本，不要求添加专用词表 token。读出位于候选之后，因此在 causal backbone 中已经能看到候选内容；没有读取一个尚未看过候选的前置标记。

编译器不把 candidate ID、标签或 gold 写进输入。它先对完整内容分词，再寻找安全的公共 token 前缀，使 pairs 与 tree 中每个候选实际得到相同的 token 序列；尾部截断保留读出后缀，并记录截断状态。

## pairs 与 tree：条件相同，摆放不同

设公共前缀为 `P`，候选分支 `A`、`B` 已各自包含读出后缀：

```text
pairs： [P | A]    [P | B]
tree：  [P | A | B]

tree 可见性（行读取列）：
          P          A          B
P       causal       ×          ×
A         ✓        causal       ×
B         ✓          ×        causal
```

每个分支只能看到共同前缀及自身过去；前缀不能看到候选，兄弟分支互不可见。这样消除的是串联 causal list 中“后面的候选能看前面的候选”的方向性条件差异，**独立 pairs 也具备这项性质**。

位置编码与物理摆放分别处理。例如公共前缀长 3、两个分支各长 2：

```text
物理下标：0 1 2 | 3 4 | 5 6
逻辑位置：0 1 2 | 3 4 | 3 4
分支：      P  |  A  |  B
```

每个分支从同一个 prefix 长度续编 position IDs；RoPE 使用这些逻辑位置。仅修改 mask、仍沿用物理 offset，不能保持与独立 pairs 相同的位置关系。

Tree 在**一个物理 chunk 内**复用前缀的隐藏状态和 K/V 计算，多个 chunk 会重算前缀。当前后端仍是 eager/SDPA 上的 dense mask；屏蔽区域不一定被算子跳过，性能收益尚待测量。完整的 attention、位置与梯度推导见[张量教程](../learning/tensors.md)，其中的 yes/no 读出与本文的 MLP 路径分别对应两种适配器。

## loss 如何训练 head 与 LoRA

[当前配方](../../recipes/nfcorpus/qwen3_base_score_head_06b.json)的 ranking 目标是：

\[
\mathcal L=0.1\,\mathcal L_{\mathrm{BCE}}+1.0\,\mathcal L_{\mathrm{pair}},
\]

\[
\mathcal L_{\mathrm{BCE}}=\frac1{|J|}\sum_{i\in J}
\operatorname{BCEWithLogits}(s_i,\mathbf1[g_i>0]),\qquad
\mathcal L_{\mathrm{pair}}=\frac1{|R|}\sum_{(i,j)\in R}
\operatorname{softplus}(-(s_i-s_j)),
\]

`J` 只包含数据中提供了标签的候选；`R={(i,j): g_i>g_j}`，所以分级标签中的 `2 > 1` 也参与排序。没有严格等级差的 query，其 pairwise 项为 0。先对每个 query 的候选／候选对取均值，再对 query batch 取均值。Loss 不把缺失标签自行当作负例；数据准备中的 train-only 弱负例仍需按数据协议记录和解释。权重 `0.1 / 1.0` 属于这次配方，不是框架对所有任务的固定规定。

```text
BCE + pairwise loss
        │
        ├── 更新 MLP 的两层权重和 bias
        │
        └── 通过 candidate hidden states 反传
                  └── 更新 backbone 中 q/k/v/o 投影的 LoRA 参数
                      原始 backbone 权重保持冻结
```

LoRA 模式只包装 backbone，`score_head` 保持独立且可训练。本配方使用 LoRA `r=8`，backbone adapter 学习率为 `2e-5`，head 学习率为 `1e-3`。训练保留 FP32 参数和更新，以 BF16 autocast 进行计算；冻结基座权重不等于切断经过基座的梯度传播。

Tree 中各分支对公共前缀的梯度会相加；在相同输入、相同逻辑位置且无随机 dropout 等差异时，应与独立 pairs 的对应梯度近似一致。full 训练同样有代码路径；具体使用哪种更新方式由配方决定。

## 正确性检查与训练结果

| 检查 | 最大绝对差 |
|---|---:|
| pairs / tree 前向分数 | `1.67e-6` |
| pairs / tree 全部可训练参数梯度 | `4.02e-6` |
| tree 候选换序、按 ID 对齐 | `1.43e-6` |
| 新进程重载的对应分数 | `0` |

记录也确认 head 与 LoRA 均获得非零梯度。这些结果支持该输入上的前向、反传和保存重载一致性；不代表全部输入、所有精度或算子的等价性，更不证明任务质量改善。新 head 的零步表现只是初始化参照，需要真实训练和留出集评价。

真实任务训练已完成 648 次优化更新，并按完整 dev 选择 best。完整 test 的 nDCG@10 为 0.290978，低于本轮原始 reranker 基线 0.354783。项目状态见 [状态页](../project/status.md)。
