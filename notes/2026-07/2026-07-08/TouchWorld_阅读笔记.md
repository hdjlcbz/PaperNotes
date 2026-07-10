<!-- arxiv: 2607.07287 -->
<!-- venue: Tech Report 2026 -->
<!-- tags: 触觉, WAM, 世界模型, VLA, 机器人操作 -->

# TouchWorld: A Predictive and Reactive Tactile Foundation Model for Dexterous Manipulation

> **论文信息**
> - 作者：Jianyi Zhou, Feiyang Hong, Yunhao Li, Yicheng Zhao, Yongjue Cen, Zirui Liu, Jiakang Huang, Zirui Chen, Ruiyang Zhang, Weizhuo Zhu, Xuhua Song, Shuo Yang
> - 通讯作者：Shuo Yang (HIT Shenzhen / PHANES AI)
> - 投稿方向：Tech Report
> - arXiv ID：2607.07287
> - 项目：https://phanes-lab.github.io/TouchWorld-website/
>
> 本文基于以下本地材料整理：
>
> - 论文 TeX 源码：`arXiv-2607.07287v2/`（主文件：`paper.tex`）
> - 论文插图：`figures/*.pdf/png`（6 张图）
> - 本文图片导出目录：`assets/touchworld/`

---

## 一、核心问题

灵巧操作需要双重能力：预测（anticipation）——接触会如何演化；反应（reaction）——偏差时快速纠正。现有触觉策略将触觉当作另一个低频率观测流，把语义推理、动作生成、接触反馈耦合在单一模型中——三个不同时间尺度的任务被迫竞争模型容量。

> TouchWorld 是一个预测+反应的层次化触觉基础模型，将触觉同时作为预测信号（tactile subgoal）和快速反馈信号（residual correction）。

![图1：TouchWorld 总览](assets/touchworld/teasor.jpg)

*图1：TouchWorld 概念总览。High-Level Planning Layer（Subtask Planner + Tactile World Model）→ Visuo-Tactile Goal-Conditioned Policy → Tactile-Conditioned Refinement Policy。触觉同时用于预测（生成 subgoal）和反馈（在线残差修正）。*

---

## 二、架构：多时间尺度层次

![图2：架构](assets/touchworld/3-layer.jpg)

*图2：TouchWorld 三层架构。每层运行在不同时间尺度——语义层最慢，策略层中速，触觉 refinement 最快。*

| 层 | 频率 | 组件 | 功能 |
|---|:---:|------|------|
| **L1: Planning** | 慢 | Subtask Planner + Tactile World Model | 语义任务分解 + 预测 visual-tactile subgoal |
| **L2: Action** | 中 | Visuo-Tactile Goal-Conditioned Policy | 从 subgoal + 多模态观测生成名义动作块 |
| **L3: Refinement** | 快 | Tactile-Conditioned Refinement Policy | 基于近期触觉/本体感知进行在线残差修正 |

**L1 → L2 → L3 的信息流**：

$$\ell_t^{\mathrm{sub}} = \pi_{\mathrm{subtask}}(\ell, \mathcal{I}_t, m_t) \quad \text{(语义子任务)}$$
$$g_t = \pi_{\mathrm{world}}(\ell, \ell_t^{\mathrm{sub}}, \mathcal{I}_t, \mathcal{X}_t) \quad \text{(触觉 subgoal)}$$
$$\hat{\mathbf{A}}_{t:t+H-1} = \pi_{\mathrm{goal}}(\ell, \ell_t^{\mathrm{sub}}, g_t, \mathcal{I}_t, \mathbf{s}_t, \mathcal{X}_t) \quad \text{(名义动作)}$$
$$\tilde{\mathbf{A}}_{\tau:\tau+W-1} = \pi_{\mathrm{tactile}}(\hat{\mathbf{A}}_{\tau:\tau+W-1}, \mathbf{s}_{\tau-k:\tau}, \mathcal{X}_{\tau-k:\tau}) \quad \text{(残差修正)}$$

**触觉的双重角色**：Tactile World Model 预测 subgoal（预测路径），Refinement Policy 用实时触觉纠正偏差（反应路径）。

---

## 三、实验

### 3.1 设置

![图3：硬件与任务](assets/touchworld/hardware.jpg)

6 个长时序灵巧操作真机任务，覆盖双场景（clean + human perturbation）：

| 任务 | 接触类型 |
|------|---------|
| Water Flower | 精细力控制 |
| Tabletop Clearing | 多物体大范围操作 |
| Cup Insertion | 紧密约束对齐 |
| Power Plug Insertion | 力反馈定位 |
| Pot Wiping | 持续接触运动 |
| Tissue Pulling | 柔顺力+变形预测 |

### 3.2 核心结果

| 设置 | TouchWorld | 最强 Baseline | 提升 |
|------|:--------:|:----------:|:---:|
| Clean | **65.0%** | 49.3% | **+15.7pp** |
| Human Perturbation | **53.7%** | 35.2% | **+18.5pp** |

> 在扰动的子集上提升更大（+18.5pp），说明 Tactile Refinement 在接触被打破后快速恢复的能力是最关键的贡献。

---

## 四、关键洞察

1. **预测+反应是触觉的正确用法**：预测告诉你应该感受什么接触，反应纠正实际接触与预测的偏差。两者缺一不可。

2. **时间尺度分离是关键**：慢的语义层（~1Hz）、中的动作层（~15Hz）、快的触觉层（~60Hz）——三者各司其职，不竞争模型容量。

3. **触觉 subgoal 是创新的中间表示**：Tactile World Model 生成的 subgoal 不是最终动作，而是一个"期望接触状态"——这使策略能够在接触状态空间中推理而非原始动作空间。
