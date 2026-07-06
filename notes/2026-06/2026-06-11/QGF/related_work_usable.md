# QGF — Related Work 可用版本

## 引用信息

- **论文**: Zhou et al., "Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning", arXiv 2606.11087, 2026.
- **可引用场景**: 使用 test-time compute / critic guidance 做 policy improvement 的 VLA / 机器人学习工作

## 该工作的贡献（客观陈述）

QGF proposes a test-time RL method for flow matching policies. The key idea is to guide the denoising process of a BC-trained flow policy with critic gradients at inference time, avoiding the instability of actor-critic training. The core technical contribution is a low-variance gradient estimator that:
(1) uses a single Euler integration step to approximate the denoised action, and
(2) replaces the Jacobian of the denoising map with the identity matrix (J ≈ I).
This estimator is cheaper than backpropagation through time and more reliable than taking critic gradients at noisy (out-of-distribution) actions. QGF achieves strong performance on OGBench offline RL benchmarks, outperforming prior test-time methods and competing with the best training-time baselines.

## 和我的工作的区别

| 维度 | QGF | 我的工作 |
|------|-----|---------|
| **任务域** | Simulation offline RL (OGBench) | 真实机器人 contact-rich manipulation |
| **策略类型** | Flow matching policy (单模态) | VLA + flow matching + 触觉 |
| **优化目标** | Single-task / goal-conditioned reward | 多模态感知下的 contact-rich task completion |
| **Critic 使用** | Test-time action guidance | 潜在的 test-time refinement + 触觉反应 |
| **传感器** | 纯仿真状态 | 视觉 + 触觉 + 本体感觉 |

## 可引用的角度

1. **作为 test-time guidance for flow policies 的代表性方法**：引用 QGF 说明梯度引导的有效性和 scaling 性质。
2. **作为 BC + critic guidance 范式**：QGF 证明训练和推理可以完全解耦，为我们的 VLA test-time refinement 思路提供支持。
3. **作为 critic gradient estimator 设计参考**：如果我们的方法也涉及 critic gradient guidance，QGF 的单步 Euler 近似可以作为 baseline。

## 英文草稿段落（可直接改写）

> Recent work has explored test-time policy optimization as an alternative to training-time actor-critic methods for expressive generative policies such as flow and diffusion models. Zhou et al. (2026) proposed Q-Guided Flow (QGF), which uses critic gradients to guide the denoising process of a behavior cloning flow policy at test time, without any additional policy training. By approximating the denoised action via a single Euler step and replacing the Jacobian with identity, QGF achieves low-variance gradient estimation that outperforms both backpropagation through time and naive gradient guidance at noisy actions. While QGF demonstrates strong results on offline RL benchmarks, it has not been applied to real-world robot manipulation tasks or multi-modal sensor settings, which is the focus of our work.

## 引用注意事项

- 不要过度吹捧（QGF 没有 real robot 实验）
- 清楚指出 QGF 的 limitation（sim only, 依赖 base BC quality）
- 与我们的工作形成互补而非竞争关系
