# Multi gate Mixture of expert Model with synthetic data Analysis

Unlike the MoE design, which has a single gating network for the entire model, the MMoE architecture includes a separate gating network for each task. 
As a result, the model can learn not just a per-sample weighting but also a per-task weighting for each of the expert networks. As a result, the MMoE can learn to represent the connections between various jobs. The gating networks for each task will learn to use various expert networks as a result of tasks having little in common.

This Analysis based on MMoE architectures on synthetic data-sets with varying levels of task correlation. I have done some comparision with various different models. 
