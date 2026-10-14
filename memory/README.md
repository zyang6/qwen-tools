# 显存分析

Qwen 单卡显存估算：Megatron / FSDP 训练，以及 MoE 推理。

在线：

- [Megatron 显存建模](https://zyang6.github.io/qwen-tools/memory/megatron.html)
- [FSDP 显存建模](https://zyang6.github.io/qwen-tools/memory/fsdp.html)
- [推理显存建模](https://zyang6.github.io/qwen-tools/memory/infer.html)

源码目录即本文件夹。GitHub：[zyang6/qwen-tools](https://github.com/zyang6/qwen-tools)。

## 选哪一页

| 页面 | 场景 | 静态怎么切 | 高峰 |
|---|---|---|---|
| [`megatron.html`](megatron.html) | Megatron actor update | `N_rank × 4`（bf16 权重+梯度）。优化器 `12×(N_dense/(TP×PP×DP_dense) + N_expert/(EP×PP×DP_expert))`，Adam 在 CPU 则为 0 | 词表 CE：`2×T×(V/TP)×4` |
| [`fsdp.html`](fsdp.html) | FSDP 全量微调 | `N/P × 12`（fp32 权重 + Adam；不含梯度） | 梯度 `N/P×4` + `6U` + 激活 |
| [`infer.html`](infer.html) | MoE 推理部署 | 权重按 EP/TP 切 | KV / 激活 / HCCL / ACLGraph / CANN / OS |

Megatron 的 N 不按卡数 P 均分，按 TP/PP/EP/ETP。FSDP 才是 `÷P`。推理页嵌入 [Eco-Sphere Memory Planner](https://eco-sphere.github.io/infer-memory-visualizer/)，权重从 safetensors header 逐张量求和。

## 使用方法

1. 从总入口打开对应页面，或直接进上面的链接。
2. 选 Qwen 模型，或填 ModelScope 仓库 `owner/name` 回车；结构从 `config.json` 拉。
3. 改切分、序列、offload / recompute，右侧实时出单卡 GiB（`1024³`）。
4. Dense 没有专家：EP=1。MoE 默认 TP=2、EP=8、P=8。ETP 固定为 1，页面不配置。P 须被 `TP×PP` 和 `EP×PP` 整除，否则页面标为非法，不给出配置。

页面右上角 **GitHub** 看源码，**反馈** 开 Issue（标题已带 `[memory/megatron]` 或 `[memory/fsdp]`）。推理页的按钮指向 Eco-Sphere 仓库。

## Megatron 口径

```text
N        = Text + ViT
N_expert = L × (gate_up + down)     # dense 为 0
N_dense  = N − N_expert
N_rank   = N_dense/(TP×PP) + N_expert/(EP×PP)    # ETP 固定为 1
P 被 TP×PP 整除               DP_dense  = P/(TP×PP)
P 被 EP×PP 整除               DP_expert = P/(EP×PP)
优化器   = 12 × (N_dense/(TP×PP×DP_dense) + N_expert/(EP×PP×DP_expert)) / 1024³
                                      # Adam CPU → 0
静态     = N_rank × 4 / 1024³ + 优化器
T        = mbs × (prompt + response)
动态内存 = 2 × T × (V/TP) × 4 / 1024³
激活值   = 1 × T × H × 2             # full recompute，只留一层入口
         = L × T × H × 2             # 关掉 recompute，全部层入口
HCCL     = n_comm × BUFFSIZE × 2 / 1024
整卡     = 静态 + 动态 + 激活 + HCCL + 碎片1 + PTA1 + Device OS 3.5
```

`n_comm = 1(DP) + [TP>1] + [PP>1] + [EP>1] + [ETP>1]`。高峰不加 MoE permute。param/grad offload 只改步间，不改高峰。

## FSDP 口径

```text
静态 = N/P × 12          # fp32 参数(4) + Adam(8)；梯度不算进静态
高峰再叠：梯度 N/P×4 + 动态内存 + 激活 + HCCL/碎片/PTA/OS
动态内存：wrap=experts 时为 6U（GMM 与 U 同形）。
wrap=decoder layer 时为 all-gather + copy_out + prefetch + 2×reduce + GMM，
前四项按高峰那一层 U_layer，GMM 只加 U_moe = (gate_up + down) × 2。
有连续 GDN 时高峰是 5×U_gdn + U_moe。
```

## 推理口径

嵌入 Eco-Sphere 工具，计算在对方页面完成：

```text
单卡 = 权重 + KV/Index Cache + 激活 + HCCL
     + ACLGraph + CANN/PTA/算子 + Device OS
```

说明与贡献见 [infer-memory-visualizer README](https://github.com/Eco-Sphere/infer-memory-visualizer/blob/main/README.md)。

## 本目录文件

```text
memory/
├── README.md        # 本文件
├── megatron.html    # Megatron actor update
├── fsdp.html        # FSDP 全量微调
└── infer.html       # 推理（iframe）
```

## 反馈

发现公式、切分或模型解析有问题，请开 [Issue](https://github.com/zyang6/qwen-tools/issues/new)，附：

- 页面（Megatron / FSDP）
- 模型与切分
- 工具结果 / 预期 / 日志或 pickle 来源
