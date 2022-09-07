# Multi gate Mixture of expert Model with synthetic data Analysis

Unlike the MoE design, which has a single gating network for the entire model, the MMoE (https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) architecture includes a separate gating network for each task. 
As a result, the model can learn not just a per-sample weighting but also a per-task weighting for each of the expert networks. As a result, the MMoE can learn to represent the connections between various jobs. The gating networks for each task will learn to use various expert networks as a result of tasks having little in common.

This Analysis based on MMoE architectures on synthetic data-sets with varying levels of task correlation. I have done some comparision with various different models. 

Another Multi-task-learning model architecture is proposed by Umberto, Michelucci, Francesca Venturini in paper "Multi-Task Learning for Multi-Dimensional Regression: Application to Luminescence Sensing"(https://www.mdpi.com/2076-3417/9/22/4748)

some of result and analysis part is not present here due university restriction.

This code bulid by Anurag Trivedi and project offered by faculty of Computer science and Chair of Process informatics and Machine Data Analysis.
feel free reach out to me if you have any Questions on my email aanuragtrivedi007@gmail.com
