# QWEN 小工具

训练工程工具入口。

## 显存分析

- [Megatron GRPO 显存建模](megatron-grpo.html)
- [FSDP 显存建模](fsdp.html)（静态 N/P×12 + 4.1 的 6×(U/EP) + 4.2 堆积 L×2U/P + 激活 + 运行时预留）
- [推理显存建模](infer.html)（嵌入 [Eco-Sphere Memory Planner](https://eco-sphere.github.io/infer-memory-visualizer/)）

后续模块：数据、性能、算子、已知问题、技术报告。
