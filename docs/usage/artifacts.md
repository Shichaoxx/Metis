# 运行产物、导出与恢复


```text
runs/<project>/<timestamp-uuid>/
  run.json                     # 状态与产物位置
  config.resolved.json          # 解析后的完整配置
  environment.json
  data.manifest.json
  logs/events.jsonl            # 训练、评估损失、checkpoint事件
  checkpoints/checkpoint-*/    # 模型/优化器/scheduler/RNG/Trainer状态
  checkpoints/index.json
  validation/                 # initial 参照及每个 epoch 的完整 dev
  sampling.json / sampling.jsonl
  parameter_updates.json       # 实际参数更新审计
  selection/best-step-N/       # checkpoint 恢复依赖的不可变历史选优
  exports/final/               # 最后训练状态
  exports/best/                # 本轮完整 dev 选出的可用产物
    score_head.safetensors     # ScoreHead family 的独立 MLP 权重
    artifact.json              # 文件清单、SHA-256、任务与来源
    model_spec.json            # 布局、基座、版本、读出、依赖
    input_spec.json             # tokenizer/compiler契约
    selection.json             # final 或完整验证集选优的明确依据
    model/                     # 完整权重或LoRA adapter
    tokenizer/
```

`cometa train CONFIG --resume CHECKPOINT` 恢复真实训练状态，并检查模型、输入、目标及数据身份。`cometa export --artifact ... --output ...` 校验并复制**已导出的预测产物**，不把任意 checkpoint 转换成模型。full export 自包含；LoRA export 依赖记录的基座（远端固定 revision，本地验证权重哈希）。LoRA 的可恢复 Trainer checkpoint 目前保存完整 wrapper state，因此大于 adapter export。


