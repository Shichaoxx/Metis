# BoolQ：小训练集拟合诊断

本实验检查 Qwen3-0.6B-Base + MLP + LoRA 能否拟合少量真实标签。它针对前两轮 BoolQ 实验中学习不足的现象，不评价泛化能力，也不改变 Metis v1 架构。

## 数据与预算

只从[第二轮读出实验](boolq-readout-selection.md)的 1,536 条训练记录中选样。按实际编译结果排除截断样本，再以固定哈希顺序选取 16 条肯定、16 条否定记录，保持 passage 唯一。开发集与预留评测记录不参与选样、训练或指标计算。来源文件的完整性校验不构成对其他划分的质量评估。

使用原 instruction、pairs/SDPA、最长 256 token、末尾 Relevance 读出和宽度 256 的两层 GELU MLP。LoRA 为 r8/alpha16，作用于 q/k/v/o，dropout 为零；基座原始参数冻结。两臂共用 seed 42、初始化及数据顺序。MLP 输出层权重初始化为零、bias 为平衡训练子集先验的 logit，即零；初始肯定概率为 0.5。

| 设置 | 原学习率档 | 较大学习率档 |
|---|---:|---:|
| LoRA learning rate | 0.00002 | 0.0002 |
| Head learning rate | 0.0002 | 0.001 |
| 最大 optimizer updates | 240 | 240 |
| Microbatch / effective batch | 4 / 8 | 4 / 8 |
| Warmup | 前 24 次更新 | 前 24 次更新 |
| 后续调度 | 恒定学习率 | 恒定学习率 |
| Weight decay / gradient clip | 0 / 1.0 | 0 / 1.0 |
| 精度 | FP32 参数与优化器，BF16 autocast | FP32 参数与优化器，BF16 autocast |

每 20 次实际更新后，在 eval 模式下完整计算 32 条训练记录的指标。连续两次检查均满足 accuracy ≥ 31/32、两类 recall 各 ≥ 15/16、NLL ≤ 0.15 时提前停止。初始评估和产物重载不计入连续两次停止检查。

这是一项优化强度诊断。两臂同时改变 LoRA 与 head 学习率，不能把差异单独归因于其中一组。样本量、类别比例、weight decay、调度和重复暴露次数也与第二轮不同，不能把跨实验差异视为单因素因果证据。

## 审计与结果解释

记录初始与每次检查的混淆矩阵、两类召回、NLL、Brier、AUROC 和预测分布；分别记录 LoRA/head 的裁剪前梯度及实际参数变化，验证冻结基座保持不变。零输出权重会使第一步到隐藏表示及 LoRA 的梯度为零，这是初始化的预期行为。

每臂导出最终 LoRA、评分头、训练预测与固定训练参考输入，并在新进程重载验证。产物属于独立研究格式，尚未接入生产 Predictor。

通过停止条件只说明这个无截断、平衡的小训练子集可以被拟合。未通过也不能单独归因于评分头；输入、初始化、优化预算和适配范围仍可能影响结果。任何后续训练配方选择都需使用独立开发数据；本实验不能作为 BoolQ benchmark 或架构升级的质量证据。

## 运行

准备第二轮的冻结训练记录、协议与相同的本地基座。安装 `train`、`peft` 可选依赖后，先在 CPU 上固定诊断协议：

```bash
python examples/research/diagnose_boolq_training.py prepare \
  --round2 runs/boolq/readout-selection \
  --model /path/to/Qwen3-0.6B-Base \
  --output runs/boolq/training-diagnostic
```

每个训练与重载命令使用独立进程。两臂分别从相同基座和初始化开始：

```bash
for arm in reference stronger; do
  python examples/research/diagnose_boolq_training.py train \
    --protocol-dir runs/boolq/training-diagnostic --arm "$arm" --device cuda:0
  python examples/research/diagnose_boolq_training.py verify \
    --export "runs/boolq/training-diagnostic/$arm/final" --device cuda:0
done
```

准备阶段记录来源协议、训练记录、token 输入及执行源码的哈希。已冻结目录不可覆盖；修改诊断设计需新建运行。每臂的 `training_audit.json`、`gradient_audit.jsonl` 和 `reload_verification.json` 分别记录训练结果、梯度和重载验证。数据、权重与原始报告不随源码分发。

## 结果

2026-10-09，两臂均完成真实训练并达到停止标准。下表全部指标来自参与训练的同一批 32 条记录。

| 学习率档位 | 首次满足条件的更新数 | 连续两次满足后停止 | 训练 Accuracy | 两类 Recall | 训练 NLL | 训练 AUROC |
|---|---:|---:|---:|---:|---:|---:|
| 原学习率档 | 80 | 100 | 100% | 各 100% | 0.000257008 | 1.000 |
| 较大学习率档 | 40 | 60 | 100% | 各 100% | 0.000001703 | 1.000 |

两臂初始 accuracy 均为 50%、NLL 为 0.693147、AUROC 为 0.5。原档位在第 40 次更新已正确判断 31/32 条，但 NLL 仍为 0.605986；仅看分类准确率会过早判定诊断完成。最终停止分别相当于对 32 条样本重复训练 25 和 15 个 epoch。

两臂共完成 160 次非零学习率更新，LoRA 与 head 均实际变化，冻结基座哈希保持一致。初始化参数和预先固定的数据顺序相同；原档位 82% 的更新触发全局梯度裁剪，较大档位为 56.7%。这项统计不能证明裁剪是前两轮学习不足的原因。

两个最终产物均在独立进程重载，对各自三条固定训练参考输入的最大 logit 差为 0，重载参数哈希与导出一致。峰值 PyTorch allocated memory 均约 2.489 GiB；单臂记录的训练与导出时间约 93 秒和 60 秒，不作为受控吞吐比较。六项离线 CPU 契约测试通过，覆盖选样、输入身份、训练顺序、AUROC 与停止条件。

本结果排除了“该结构和当前训练链路完全不能拟合这 32 条样本”的判断。它没有证明模型会泛化，也没有证明较大学习率可修复完整 BoolQ 任务。前两轮全肯定预测仍需从任务数据、优化过程和泛化差距继续诊断；本轮不读取或重跑旧预留评测集，不支持发布新的架构质量提升声明。

协议 SHA256：`e984b1de0fd52fea41a9261b4d719c2883d24f8ff0e9c77ffdf2cc52171ef0e4`。
