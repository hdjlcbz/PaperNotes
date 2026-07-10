<!-- arxiv: 2607.02840 -->
<!-- venue: CoRL 2026 -->
<!-- tags: WAM, 触觉, VLA, 世界模型, 强化学习 -->

# TACO: TActile World Model as a Self-COrrector for Scalable VLA Post-Training

> **论文信息**
> - 作者：Yuxing Qin, Shuai Tian, Yupeng Zheng, Yuhang Zheng, Kechun Xu, Ce Hao, Yujie Zang, Weize Li, Haoran Li, Wenchao Ding, Dongbin Zhao
> - 通讯作者：Haoran Li, Wenchao Ding (CASIA / TARS Robotics)
> - 投稿方向：CoRL 2026（preprint）
> - arXiv ID：2607.02840
>
> 本文基于以下本地材料整理：
>
> - 论文 TeX 源码：`arXiv-2607.02840v1/`（主文件：`main.tex`，按 `sec/` 分章节）
> - 论文插图：`image/*.pdf/png/jpg`（14 张图）
> - 本文图片导出目录：`assets/taco/`

---

## 一、核心问题

VLA 模型在接触-rich 操作中的失败往往是局部的（如对齐偏差、接触力不足），而非任务级语义错误。这些失败难以从视觉检测，但触觉反馈可以捕获。

问题是：如何**规模化**地利用触觉反馈来纠正 VLA 策略？靠人工干预太贵，而纯视觉世界模型可能产生"看起来很对但接触不对"的 imagined rollout。

> TACO 用触觉感知世界模型实现了规模化 VLA post-training：Recognize → Imagine → Label 循环，将真实失败转化为触觉纠正监督。

![图1：TACO 框架](assets/taco/pipeline_v5.jpg)

*图1：TACO 框架。(1) Recognize：部署策略收集 rollout，用 progress model 识别失败临界状态；(2) Imagine：visuo-tactile 世界模型从失败状态生成局部纠正片段；(3) Label：progress-action model 为纠正片段标注可执行动作。Knowledge-Insulated 触觉适配 + advantage-conditioned training 完成 VLA post-training。*

---

## 二、核心方法

### 2.1 触觉感知世界模型

![图2：世界模型架构](assets/taco/model_arch_v3.jpg)

*图2：触觉感知世界模型。左：visuo-tactile joint denoising——Wan2.2-TI2V-5B 基础上联合去噪视频+力信号（12 维，双指 6-DoF）；右：统一 progress-action model——DINOv2 视觉通路 + MLP 触觉通路，联合预测纠正动作和任务进度。*

| 组件 | 功能 |
|------|------|
| Visuo-Tactile Generation | 联合去噪视频 latent + 12D 力轨迹，temporal RoPE 对齐 |
| Progress-Action Model | DINOv2 视觉 + MLP 力感知 → 预测 $\hat{a}_t$（7D 动作）+ $\hat{p}_t$（任务进度） |

**关键设计**：
- **Temporal RoPE Alignment**：力 token 对齐到视频 latent 时间轴
- **First-Frame Force Anchor**：保持 $F_0$ 干净作为参考锚点，减少接触状态歧义

### 2.2 Recognize-Imagine-Label 循环

1. **Recognize**：progress model 识别 rollout 中进度停滞/下降的"失败临界状态"
2. **Imagine**：从临界状态出发，世界模型想象局部 visuo-tactile 纠正片段
3. **Label**：progress-action model 为想象片段标注纠正动作

### 2.3 Knowledge-Insulated 触觉适配

直接端到端 post-training 可能破坏 VLA 的预训练视觉-语言先验。TACO 采用两层保护：

- **Tactile Adapter**：仅训练轻量触觉 adapter 而非全模型，保护视觉-语言知识
- **Advantage-Conditioned Training**：条件于 advantage（当前 vs 纠正后的状态价值差），鼓励策略模仿高 advantage 的纠正动作

---

## 三、实验与结果

### 3.1 任务设置

6 个真机接触-rich 任务：Insert Flower, Wipe Whiteboard, Twist Bottle Cap, Play Xylophone, Toast Bread, Move Hanoi Rings。

基于 π₀.₅ 作为 base policy，两轮迭代 post-training。

### 3.2 核心结果

| 方法 | SR | Contact Steps |
|------|:--:|:------------:|
| Base Policy | 0.38 | 185.5 |
| Iter1: Filtered BC | 0.41 | 148.8 |
| Iter1: TACO (w/o KI) | 0.49 | 154.8 |
| Iter1: **TACO** | **0.66** | 141.8 |
| Iter2: **TACO** | **0.70** | 131.0 |

> Base → Iter2 TACO: **+32% 绝对提升**；KI 消融（w/o KI）仅 0.49，证明知识绝缘适配至关重要。

### 3.3 关键消融

- **w/o KI** (直接端到端 post-training)：SR 从 0.66 → 0.49，验证了保护预训练先验的必要性
- **w/o Tactile** (仅视觉世界模型)：想象纠正的接触一致性差，纠正动作不可靠
- **迭代收益**：第一轮 +28%，第二轮 +4%，收益递减但仍有效

---

## 四、关键洞察

1. **触觉世界模型作为"自我纠正器"**：不是替代策略训练，而是将真实失败转化为触觉纠正数据，实现闭环改进。

2. **KI 既简单又关键**：仅冻结 VLA weights + 加轻量触觉 adapter 就保护了预训练先验。去掉 KI，想象纠正的优势几乎消失。

3. **Progress 检测比失败检测更有效**：识别"进度停滞"状态（而非等任务完全失败后再纠正）——这在长时序操作中意味着更丰富的纠正机会。

4. **Temporal RoPE + Force Anchor**：解决视频-力联合去噪中的模态对齐问题，是一个细致的工程创新。
