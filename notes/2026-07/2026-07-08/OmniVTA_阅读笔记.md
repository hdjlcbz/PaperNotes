<!-- arxiv: 2603.19201 -->
<!-- venue: IEEE T-RO 2026（投稿中） -->
<!-- tags: 触觉, WAM, 世界模型, 机器人操作, 表征学习 -->

# OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation

> **论文信息**
> - 作者：Yuhang Zheng, Songen Gu, Weize Li, Yupeng Zheng, Yujie Zang, Shuai Tian, Xiang Li, Ce Hao, Chen Gao, Si Liu, Haoran Li, Yilun Chen, Shuicheng Yan, Wenchao Ding
> - 通讯作者：Yupeng Zheng, Shuicheng Yan, Wenchao Ding (TARS Robotics / NUS / CASIA)
> - 投稿方向：IEEE Transactions on Robotics
> - arXiv ID：2603.19201
> - 项目：https://mrsecant.github.io/OmniVTA
>
> 本文基于以下本地材料整理：
>
> - 论文 TeX 源码：`arXiv-2603.19201v2/`（主文件：`main.tex`，按 `sections/` 分章节）
> - 论文插图：`fig/*.pdf`（16 张图）
> - 本文图片导出目录：`assets/omnivta/`

---

## 一、核心问题

接触-rich 操作（擦拭、削皮、切割、装配等）需要精确感知接触力、摩擦变化和状态转移——这些是纯视觉无法可靠推断的。尽管 visuotactile 操作研究日益增长，两个瓶颈持续存在：

1. **数据瓶颈**：现有 visuotactile 数据集规模小、任务覆盖窄
2. **方法瓶颈**：现有方法将触觉信号视为被动观测，而非主动建模接触动力学或支持闭环控制

> OmniVTA 同时解决这两个瓶颈：构建了 OmniViTac 大规模数据集（21,000+ 轨迹），并提出了基于世界模型的 visuotactile 操作框架。

![图1：OmniVTA 总览](assets/omnivta/teaser.jpg)

*图1：OmniVTA 总览。(左) OmniViTac——21,000+ 条 visuotactile-action 对齐轨迹，86 个任务，按 6 种物理交互模式组织；(中) OmniVTA 框架——触觉 VAE + visuotactile 世界模型 + 自适应融合策略 + 60Hz 反射控制器；(右) 真机实验在所有 6 个类别上超越 baseline。*

---

## 二、OmniViTac 数据集

![图2：数据集概览](assets/omnivta/dataset_teaser_0318.jpg)

*图2：OmniViTac 数据集覆盖 100+ 物体、86 个接触-rich 任务，按 6 种物理交互模式分类——按压、滑动、剥离、切割、装配、抓取。每类包含多个物体和多种交互条件。*

| 维度 | 规模 |
|------|------|
| 轨迹数 | 21,000+ |
| 任务数 | 86 |
| 物体数 | 100+ |
| 交互模式 | 6（Wipe/Peel/Cut/Assembly/Grasp/Adjustment） |
| 数据采集 | TacUMI + xArm7, 15Hz 视觉 + 60Hz 触觉 |

两个结构性质指导了 OmniVTA 架构设计：
- **空间局部性**：触觉信号在空间上高度局部化——接触点周围变化大，远处几乎不变
- **接触驱动动力学**：触觉信号的变化主要由接触状态变化驱动，非接触阶段信号几乎静止

---

## 三、OmniVTA 方法

### 3.1 系统架构

![图3：系统架构](assets/omnivta/system.jpg)

*图3：OmniVTA 的分层 slow-fast 架构。Slow Policy 包含 visuotactile 世界模型 + 自适应融合策略，从多模态输入规划长时序动作块；Fast Policy 在 60Hz 基于触觉反馈输出精细修正动作。最终动作为两者的加权求和。*

OmniVTA 将操作显式分解为 **Slow Planning**（15Hz）和 **Fast Reflexive Control**（60Hz）：

```
┌─────────────────────────────────────────────────────────┐
│                 OmniVTA Pipeline                         │
├─────────────────────────────────────────────────────────┤
│  Slow Policy (15Hz)                                      │
│  ┌──────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │ Vision   │  │ TactileVAE    │  │ Visuo-Tactile    │  │
│  │ (RGB)    │  │ (3D marker    │  │ World Model      │  │
│  │ 15Hz     │  │  displacement)│  │ (predict future  │  │
│  │          │  │               │  │  tactile states) │  │
│  └────┬─────┘  └──────┬────────┘  └────────┬─────────┘  │
│       │               │                    │             │
│       └───────────────┼────────────────────┘             │
│                       ▼                                  │
│            Adaptive Fusion Policy (gated)                 │
│            → action chunk (6 steps)                       │
│                                                          │
│  Fast Policy (60Hz)                                       │
│  ┌──────────────────────────────────────────┐            │
│  │ Reflexive Latent Tactile Controller      │            │
│  │ Δa = f(predicted_tac - observed_tac)     │            │
│  │ → corrective delta actions               │            │
│  └──────────────────────────────────────────┘            │
│                                                          │
│  a_final = a_slow + α × Δa_fast                          │
└─────────────────────────────────────────────────────────┘
```

### 3.2 TactileVAE

![图4：TactileVAE](assets/omnivta/vae.jpg)

*图4：TactileVAE 使用因果 3D 卷积编码器 + 隐式神经表示（INR）解码器。编码器将 3D marker displacement 压缩为时空特征图；INR 解码器将特征图建模为连续形变场，支持任意分辨率查询重建。*

关键设计：
- **3D marker displacement 作为输入**：替代高分辨率触觉图像，捕捉接触形变同时保持低分辨率（如 Xense: 35×20×3）
- **因果 3D 卷积**：时序轴上 causal，保证实时部署一致性
- **INR 解码器**：建模连续形变场 $\mathbf{d}(\mathbf{x}) = \mathcal{D}_{\theta}(\gamma(\mathbf{x}), \Phi(\mathbf{z}_t, \mathbf{x}))$，支持任意点查询

### 3.3 Visuo-Tactile World Model

![图5：Slow Policy——世界模型 + 融合策略](assets/omnivta/slow-policy.jpg)

*图5：(a) 双流 visuotactile 世界模型——视觉和触觉分支使用独立的时空扩散 Transformer，通过共享多模态条件编码器对齐；(b) 自适应融合策略——Latent Tactile Differential (LTD) 编码器 + gating 机制，自适应平衡视觉和触觉信息。*

三个核心组件：

| 组件 | 功能 |
|------|------|
| **Two-Stream WM** | 视觉+触觉双流时空扩散 Transformer，并行建模+联合生成 |
| **Multimodal Conditioner** | 联合编码视觉/触觉/动作（2D 图像平面投影），共享条件向量注入两分支 |
| **Dynamic-Aware Loss** | 基于触觉时序差分的动态权重图 + 响应幅值权重，强调高频接触变化区域 |

推理时不生成视觉观测——世界模型仅预测未来触觉信号，实现更高频率的 rollout。

### 3.4 自适应融合策略 + 反射控制器

**LTD Encoder**：编码"当前触觉 - 预测触觉"的差分信号，而非直接编码触觉原始值，让策略感知接触状态变化。

**Gating 机制**：根据接触状态自适应平衡视觉和触觉特征权重——接触阶段触觉权重高，非接触阶段视觉权重高。

![图6：反射控制器](assets/omnivta/controller.jpg)

*图6：Reflexive Latent Tactile Controller——将预测触觉作为期望接触状态，与实际触觉的 latent 差分输入控制器，输出 60Hz 修正动作。该设计能恢复接触而不产生过大接触力，保护传感器。*

> 关键数据：OmniVTA 平均切向形变 0.35（最大 0.72），而 RDP 平均 0.56（最大 1.1）——说明 RLTC 在维持接触的同时保护传感器。

---

## 四、实验与结果

### 4.1 真机操作性能

![图7：六类操作任务](assets/omnivta/manipulation.jpg)

*图7：六类接触-rich 操作任务的真机 rollout 可视化——Wipe、Peel、Cut、Assembly、Grasp、Adjustment。*

| 方法 | Wipe(O/G/P) | Peel(O/G/P) | Cut(O/G/P) | Assembly(O/G/P) | Grasp(O) | Adjust(O/G) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| DP | 12/5/0 | 6/0/0 | 28/10/0 | 10/0/5 | 20 | 0/0 |
| DP+tactile | 36/28/0 | 32/20/8 | 33/15/13 | 30/10/10 | 48 | 25/15 |
| KineDex | 40/25/0 | 24/13/5 | 38/30/20 | 30/15/15 | 65 | 30/20 |
| RDP | 50/38/42 | 48/36/45 | 65/50/43 | **60/50**/35 | **88** | 50/50 |
| OmniVTA w/o RLTC | 66/40/25 | 40/30/20 | 50/50/20 | 40/35/20 | 70 | 40/30 |
| **OmniVTA** | **80/58/60** | **55/48/63** | **85/83/60** | 60/**50/40** | **90** | **65/65** |

> 3 种评估设置（O=物体多样性, G=泛化, P=扰动鲁棒性）× 6 个任务 = 15 个指标，OmniVTA 在 13 个上排第一。

### 4.2 世界模型预测

![图8：触觉预测可视化](assets/omnivta/wm.jpg)

*图8：visuotactile 世界模型在六类任务上的预测可视化。红色箭头=预测切向形变，蓝色=真值。模型能准确预测接触区域的方向和幅值。*

- 在所有 3 个预测 horizon（8/16/24 帧）上超越所有 baseline
- 联合生成视觉特征能提升触觉预测精度（视觉提供互补的全局动力学线索）
- 2D 动作表示（末端执行器在图像平面的投影）泛化性最优
- Dynamic-Aware Loss 对高频触觉变化的预测精度有明显增益

### 4.3 组件消融

| 消融 | 关键发现 |
|------|---------|
| TactileVAE vs PCA/PointCloud AE | INR 解码器 + 位置编码对跨传感器泛化至关重要 |
| 预测触觉 vs 仅当前触觉 | 预测未来触觉状态显著提升策略性能 |
| LTD Encoder vs 简单拼接 | LTD 编码差分信号比直接拼接原始触觉更有效 |
| Gating vs 固定权重 | 自适应门控优于固定融合权重，门控权重与接触状态高度相关 |
| RLTC vs 无 RLTC | 60Hz 反射控制器在所有扰动任务上提升 ~20pp |

---

## 五、关键洞察与技术亮点

1. **触觉信号的两条结构性质**：空间局部性 + 接触驱动动力学——这两条性质直接指导了整个架构设计（TactileVAE 的 INR 解码器、Dynamic-Aware Loss）。

2. **世界模型在推理时不生成视觉**：仅预测未来触觉信号供策略和控制器使用，大幅提升推理效率。

3. **预测触觉作为"期望接触状态"**：RLTC 将预测触觉作为控制目标，实际触觉作为反馈——这种 reflexive control 范式的创新之处在于用世界模型替代了手工设计的期望接触力轨迹。

4. **2D 动作表示优于 3D**：将末端执行器位置投影到图像平面作为动作条件——与视觉观测天然对齐，泛化性更好。

5. **保护传感器**：RLTC 设计的控制增益确保不会产生过大的接触力（平均形变 0.35 vs RDP 的 0.56），避免了传感器损坏。

---

## 六、代码实现解读

无独立代码仓库。核心训练分四阶段：

| 阶段 | 组件 | 数据 | 配置 |
|------|------|------|------|
| 1 | TactileVAE | 120 万触觉样本 | 8×A100, 50 epochs |
| 2 | VT World Model | 每类 5-6 物体, 各 150 轨迹 | AdamW lr=1e-4, 100k steps |
| 3 | Fusion Policy | 同上 | 250k steps |
| 4 | RLTC | 同上 | 独立训练 |

**关键参数**：
- 视觉 15Hz, 触觉 60Hz, 本体感知 60Hz
- 策略输出 15FPS 动作块（预测 6 步，插值到 60Hz）
- 触觉输入：当前帧 + 前 7 帧（8 帧/时间窗口）
- 视觉输入：当前帧 + 前 1 帧（2 帧）
- Xense 触觉分辨率：35×20×3 marker displacement

---

## 七、局限性

1. **仅限单臂夹爪**：未涉及双臂或灵巧手
2. **世界模型规模有限**：未来可扩展到更大、更多样的数据
3. **跨具身迁移未验证**：触觉表示和世界模型能否迁移到不同机器人形态仍需探索
