# 训练配方

当前配方是 [Qwen3 Base + ScoreHead + LoRA](nfcorpus/qwen3_base_score_head_06b.json)，对应 [NFCorpus cookbook](../cookbooks/score-head-plan.md)，已完成两轮训练。

配方按任务组织，路径相对配置文件解析。已执行配方保留原值；后续实验另存配置，明确基座版本、输入布局、监督目标、采样、dev 选优和输出位置。训练保存 resolved config、环境和数据身份，test 用于最终报告。
