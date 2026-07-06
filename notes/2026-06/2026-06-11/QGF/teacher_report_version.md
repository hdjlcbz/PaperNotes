<!-- arxiv: 2606.11087 -->
<!-- venue: arXiv 2026（投稿中） -->
<!-- tags: 强化学习, 扩散模型, 离线RL, 视频生成 -->

# QGF — 老师汇报版

## 论文信息

- **标题**: Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning
- **作者**: Zhiyuan Zhou, Andy Peng, Charles Xu, Qiyang Li, Tobias Springenberg, Kevin Frans, Sergey Levine (UC Berkeley & Physical Intelligence)
- **代码**: github.com/zhouzypaul/qgf
- **发表**: arXiv 预印本，2026

## 重要性评估：★★★（非常重要）

**为什么重要**：这篇论文解决了一个核心问题——如何把 RL 的 reward optimization 能力嫁接到 flow/diffusion policy 上，同时避免 actor-critic 训练的灾难性不稳定。它用的是 test-time guidance 的范式，对我们做 VLA + flow matching + RL 的组合非常关键。

## 是否撞车：不直接撞车，但方向高度重叠

QGF 是在 simulation benchmark（OGBench）上做的，不涉及真实机器人、触觉、VLA。但它的技术路线（BC flow policy + critic gradient guidance）和我们未来要做的事情（触觉 VLA + test-time refinement）在方法论上有很大交集。我们需要区分：我们处理的是多模态感知下的接触丰富操作，QGF 只处理 simulation 控制任务。

## 核心 idea（3-5 句话）

1. 策略用普通 BC 训练 flow matching，critic 用 IQL 单独训练。
2. 推理时，flow 的每个去噪步加上 critic gradient 作为 guidance：a_{t+δ} = a_t + δ * (v_θ(a_t) + 1/β * ∇Q(s, â₁))
3. 关键技巧：用单步 Euler 近似去噪动作 â₁（不用完整 ODE），然后直接用 ∇Q(s, â₁) 代替复杂的 chain rule。
4. 这个估计器比 BPTT 快、比 OOD gradient 准、比完整去噪链的方差小。
5. 所有 reward optimization 在 inference 完成，训练阶段只有稳定的 BC + TD learning。

## 我能借鉴什么

1. **test-time guidance 范式**：我们的 VLA 也可以用类似方法——BC 训练 VLA，然后用 task reward critic 在推理时引导动作优化。
2. **梯度估计器设计**：如果我们要在 flow matching VLA 上做 test-time refinement，QGF 的单步 Euler + 丢 Jacobian 是最直接可用的方案。
3. **critic 训练解耦**：QGF 证明 IQL 这样 in-sample 的 critic 就足够做 test-time guidance，不需要 bootstrapping 的复杂 critic。
4. **scaling 性质**：test-time guidance 随模型增大持续受益，这和我们 scale VLA 的趋势一致。

## 不足与风险

1. 只在 simulation benchmark 上验证，没有 real robot 实验。
2. critic 对 OOD 动作的估计偏误仍是根本限制（虽然 QGF 已经比 OOD gradient 好很多）。
3. 如果 base BC policy 太差（如训练不足），test-time guidance 也救不回来。

## 结论

这是一篇方法上非常优雅、对我们方向有直接参考价值的论文。建议重点跟进它的 test-time guidance 机制，并思考如何在我们的 VLA + 触觉框架中引入类似的 critic-guided refinement。
