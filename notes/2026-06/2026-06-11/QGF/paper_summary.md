# QGF — 精简总结

## 一句话总结

QGF 提出一种纯 test-time RL 方法：用行为克隆（BC）训练 flow matching 策略、单独训练 Q 函数，推理时用 critic gradient 引导 flow 的去噪过程生成高价值动作，完全不需要 RL 式的策略训练。

## 核心贡献

1. **新的梯度估计器**：用单步 Euler 近似去噪动作 + 丢掉 Jacobian（J ≈ I），得到一个低方差、低计算成本的 critic gradient estimator，避免 BPTT 的高开销和 OOD gradient 的偏差。
2. **纯 test-time policy optimization**：策略只用 BC 训练，所有 reward optimization 发生在推理时，完全避开 actor-critic 的不稳定性。
3. **强实验性能**：在 OGBench 单任务和 goal-conditioned RL 上超越所有 test-time 方法，与最强 training-time 方法（EDP）竞争。
4. **良好的模型规模 scaling**：QGF 随模型增大持续提升，而训练期方法（QAM）出现退化。
5. **critic 无关性**：QGF 可与 IQL critic 或 QAM bootstrapping critic 配合使用。

## 关键结果

- QGF 在 20 个 OGBench 单任务上平均性能超越 QFQL（~30%+）且接近 EDP
- 与 BFN (N=4) 相比，QGF 性能更好且 FLOPs 低数个数量级
- Scaling 实验：QGF 从 800k → 3.2M 参数性能提升 ~4×

## 和我的工作关系

QGF 研究的 test-time guidance + flow/diffusion policy 与我的 flow matching VLA 方向高度相关。它的 critic gradient estimator 方法可以直接用于 VLA 的 test-time refinement，特别是应对触觉反馈等需要高频响应的场景。但 QGF 目前只做 simulation benchmark（OGBench），不涉及真实机器人，也不涉及多模态传感器。
