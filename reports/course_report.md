# SDXL 小样本动漫人物与风格 LoRA 微调实验报告

> 当前是报告草稿。生成图片后，需要根据 `outputs/report_experiments/seed_<SEED_BASE>/contact_sheets/` 中的结果补全实验观察、表格和结论。

## 1. 项目目标

本项目研究在小样本动漫人物头像数据集上，如何使用 SDXL、PTI warmup 和 LoRA 微调得到可控的风格或人物生成能力。与只训练一个统一风格不同，我们将数据分成两类关键词：

- 人物关键词：`<eva_rei_headshot>`，用于学习 Ayanami Rei 的身份特征。
- 风格关键词：`<persona_5_headshot>` 与 `<ghibli_headshot>`，用于学习 Persona 5 式红黑图形化风格和吉卜力式柔和手绘动画风格。

报告重点不是简单证明 LoRA 可以出图，而是分析小样本 LoRA 的推理强度、训练动态、跨 prompt 泛化、多 token merge 控制和 merge 权重冲突。

## 2. 数据集与标注

当前实验使用三个数据集：

| 数据集 | 图像数量 | 训练 token | 目标 |
|---|---:|---|---|
| `data/Ghibli` | 24 | `<ghibli_headshot>` | 吉卜力感人物与场景头像 |
| `data/persona_5` | 24 | `<persona_5_headshot>` | Persona 5 红黑图形化人物风格 |
| `data/EVA_rei` | 23 | `<eva_rei_headshot>` | Ayanami Rei 人物身份 |

每张图片配有同名 `.txt` caption。caption 采用短描述，而不是堆叠通用高质量标签。这样做的原因是小数据集 LoRA 对错误 caption 较敏感，过长或虚构的 caption 容易把不存在的服装、动作、背景写入训练信号。

## 3. 技术路线

### 3.1 为什么使用 SDXL

SDXL 相比旧的 SD 1.5 有更强的基础图像质量和 prompt 理解能力，适合 1024 x 1024 分辨率人物头像生成。课程作业要求比较基础模型和微调模型时，SDXL base 也能提供更强的对照组：如果 LoRA 仍然能在 SDXL base 已经较强的情况下带来稳定风格变化，说明微调是有效的。

### 3.2 为什么使用 PTI warmup

我们不是只想训练通用风格词，而是希望模型学会新的触发词，例如 `<eva_rei_headshot>`。PTI/Textual Inversion warmup 先只优化这个新 token 在两个 SDXL text encoder 中的 embedding，让 token 初步对齐目标人物或风格概念。之后再训练 LoRA，可以减少“新 token 没有语义、LoRA 需要同时学习触发和风格”的不稳定性。

### 3.3 为什么使用 LoRA

LoRA 不直接全量更新 SDXL，而是在 UNet 和 text encoder 的部分线性层上学习低秩增量。它的优点是训练成本低、checkpoint 小、适合小数据集，并且便于后续合并多个 LoRA：

```text
W' = W + scale * B @ A
```

在多个风格实验中，LoRA 也便于单独训练、比较 checkpoint、调整推理 scale，并在最后做 merge。

### 3.4 为什么做 Multi-token Merge

常见 merge 会把多个 LoRA 合成一个新触发词，但这样会损失控制性。我们的目标是测试“人物 token + 风格 token”的组合，因此使用 `--embed-merge keep` 保留：

```text
<eva_rei_headshot>
<persona_5_headshot>
<ghibli_headshot>
```

这样同一个 merged adapter 可以测试 Rei on Persona 5 和 Rei on Ghibli，而不是只生成一个平均混合风格。

## 4. 实验参数

训练默认参数：

| 参数 | 值 |
|---|---|
| Base model | SDXL base 1.0 |
| Resolution | 1024 x 1024 |
| Batch size | 1 per GPU |
| Gradient accumulation | 4 |
| Max steps | 2000 |
| PTI warmup steps | 500 |
| Checkpoint interval | 200 |
| LoRA rank / alpha | 16 / 16 |
| UNet LoRA learning rate | `1e-4` |
| Text encoder LoRA learning rate | `5e-6` |
| PTI learning rate | `5e-4` |
| Precision | fp16 |

推理默认参数：

| 参数 | 值 |
|---|---|
| Scheduler | DPMSolverMultistepScheduler |
| Resolution | 1024 x 1024 |
| Steps | 30 |
| Guidance scale | 7.5 |
| Default LoRA scale | 0.65 |
| Seeds | 由 `SEED_BASE` 决定，例如 `SEED_BASE` 和 `SEED_BASE + 1` |

实验脚本：

```bash
SEED_BASE=1234 WIDTH=1024 HEIGHT=1024 STEPS=30 bash experiments/run_report_experiments.sh
```

## 5. 实验结果

### 5.1 原始 SDXL vs 单 LoRA

目的：比较 base SDXL 和单风格 LoRA 在相同 prompt 和 seed 下的差异。

结果图：

```text
outputs/report_experiments/seed_<SEED_BASE>/contact_sheets/base_vs_single.jpg
```

待填写观察：

| 对比 | 观察 | 结论 |
|---|---|---|
| SDXL base vs Ghibli LoRA | TODO | TODO |
| SDXL base vs Persona 5 LoRA | TODO | TODO |
| SDXL base vs EVA Rei LoRA | TODO | TODO |

### 5.2 LoRA Scale 扫描

目的：分析 `lora_scale` 对风格强度、人物稳定性和画面质量的影响。

测试：

```text
0.4 / 0.55 / 0.65 / 0.75 / 0.9
```

结果图：

```text
outputs/report_experiments/seed_<SEED_BASE>/contact_sheets/scale_sweep.jpg
```

待填写观察：

| 风格 | 推荐 scale | 观察 |
|---|---:|---|
| Ghibli | TODO | TODO |
| Persona 5 | TODO | TODO |
| EVA Rei | TODO | TODO |

### 5.3 Checkpoint 训练动态

目的：观察训练过程中风格或人物特征何时出现，是否在后期过拟合。

测试：

```text
checkpoint-200 / checkpoint-600 / checkpoint-1000 / checkpoint-1400 / checkpoint-1800 / final
```

结果图：

```text
outputs/report_experiments/seed_<SEED_BASE>/contact_sheets/checkpoint_dynamics.jpg
```

待填写观察：

| 风格 | 早期 checkpoint | 中后期 checkpoint | final |
|---|---|---|---|
| Ghibli | TODO | TODO | TODO |
| Persona 5 | TODO | TODO | TODO |
| EVA Rei | TODO | TODO | TODO |

### 5.4 跨 Prompt 泛化

目的：比较 in-domain 和 out-of-domain prompt，判断 LoRA 是否只记住训练集构图。

结果图：

```text
outputs/report_experiments/seed_<SEED_BASE>/contact_sheets/generalization.jpg
```

待填写观察：

| 风格 | In-domain 表现 | Out-of-domain 表现 | 泛化结论 |
|---|---|---|---|
| Ghibli | TODO | TODO | TODO |
| Persona 5 | TODO | TODO | TODO |
| EVA Rei | TODO | TODO | TODO |

### 5.5 Multi-token Merge 控制

目的：在一个 merged adapter 内测试人物 token 和风格 token 的组合控制。

重点 prompt：

```text
<eva_rei_headshot>, <persona_5_headshot>, Ayanami Rei ...
<eva_rei_headshot>, <ghibli_headshot>, Ayanami Rei ...
```

结果图：

```text
outputs/report_experiments/seed_<SEED_BASE>/contact_sheets/multitoken_merge.jpg
```

待填写观察：

| 组合 | Rei 身份保留 | 风格迁移 | 问题 |
|---|---|---|---|
| Rei only | TODO | TODO | TODO |
| Persona only | TODO | TODO | TODO |
| Ghibli only | TODO | TODO | TODO |
| Rei on Persona 5 | TODO | TODO | TODO |
| Rei on Ghibli | TODO | TODO | TODO |

### 5.6 Merge 权重比例

目的：测试 concat merge 中不同权重是否影响最终风格倾向。

测试：

```text
equal = 1 / 1 / 1
ghibli-heavy = 0.6 / 0.2 / 0.2
persona-heavy = 0.2 / 0.6 / 0.2
rei-heavy = 0.2 / 0.2 / 0.6
```

结果图：

```text
outputs/report_experiments/seed_<SEED_BASE>/contact_sheets/merge_weights.jpg
```

待填写观察：

| 权重设置 | Rei only | Rei on Persona 5 | Rei on Ghibli | 结论 |
|---|---|---|---|---|
| equal | TODO | TODO | TODO | TODO |
| ghibli-heavy | TODO | TODO | TODO | TODO |
| persona-heavy | TODO | TODO | TODO | TODO |
| rei-heavy | TODO | TODO | TODO | TODO |

## 6. 总体结论

待实验完成后填写。建议围绕下面问题总结：

1. 单 LoRA 是否显著优于 base SDXL？
2. 默认 `lora_scale=0.65` 是否合理？
3. final checkpoint 是否优于中间 checkpoint？
4. out-of-domain prompt 下最容易失败的是身份、风格还是画质？
5. Multi-token merge 是否能实现 Rei on Persona 5 和 Rei on Ghibli？
6. Merge 权重是否提供了可解释的控制能力？

## 7. 局限性

当前实验仍有几个限制：

- 每个数据集只有 20 多张图，容易学习到训练图构图偏好。
- 人工 caption 虽然清理过，但仍可能存在描述粒度不一致。
- Multi-token merge 的 token 控制不是严格解耦，人物和风格可能互相污染。
- 本报告主要依赖人工视觉观察，没有使用 FID/KID 等大样本指标。

