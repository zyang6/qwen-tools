# QWEN 小工具

训练工程工具入口。

## 显存分析

- [Megatron 显存建模](megatron-grpo.html)（静态 N_rank×4；高峰词表 2×T×(V/TP)×4 + MoE 项）
- [FSDP 显存建模](fsdp.html)（静态 N/P×10 + 动态 梯度×2 + 6U + 激活 + 运行时预留）
- [推理显存建模](infer.html)（嵌入 [Eco-Sphere Memory Planner](https://eco-sphere.github.io/infer-memory-visualizer/)）

后续模块：数据、性能、算子、已知问题、技术报告。
