# Mamba
> **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**\
> Albert Gu*, Tri Dao*\
> Paper: https://arxiv.org/abs/2312.00752

> **Transformers are SSMs: Generalized Models and Efficient Algorithms**\
>     **Through Structured State Space Duality**\
> Tri Dao*, Albert Gu*\
> Paper: https://arxiv.org/abs/2405.21060

## About

Mamba is a new state space model architecture showing promising performance on information-dense data such as language modeling, where previous subquadratic models fall short of Transformers.
It is based on the line of progress on [structured state space models](https://github.com/state-spaces/s4),
with an efficient hardware-aware design and implementation in the spirit of [FlashAttention](https://github.com/Dao-AILab/flash-attention).

## Installation

- [Option] `pip install causal-conv1d>=1.4.0`: an efficient implementation of a simple causal Conv1d layer used inside the Mamba block.
- `pip install mamba-ssm`: the core Mamba package.
- `pip install mamba-ssm[causal-conv1d]`: To install core Mamba package and causal-conv1d.
- `pip install mamba-ssm[dev]`: To install core Mamba package and dev depdencies.

It can also be built from source with `pip install .` from this repository.

Try passing `--no-build-isolation` to `pip` if installation encounters difficulties either when building from source or installing from PyPi. Common `pip` complaints that can be resolved in this way include PyTorch versions, but other cases exist as well.

Other requirements:
- Linux
- NVIDIA GPU
- PyTorch 1.12+
- CUDA 11.6+

For AMD cards, see additional prerequisites below.

# Latent Mamba-DQN: Improving Temporal Dependency Modeling in Deep Q-Learning via Selective State Summarization

## Overview

This repository provides the official implementation of **Latent Mamba-DQN**, a Deep Q-Learning framework enhanced with Mamba-SSM for efficient temporal dependency modeling, selective state summarization, and improved policy stability in dynamic environments.

The associated publication is currently under preparation for peer-reviewed journal submission. This repository will be updated with the DOI and publication details upon acceptance.

> **Latent Mamba-DQN: Improving Temporal Dependency Modeling in Deep Q-Learning via Selective State Summarization**  

## 1. Architecture Overview

Latent Mamba-DQN integrates a **Mamba-based State Space Model (Mamba-SSM)** into the DQN framework to effectively capture temporal information from sequential observations. The proposed model processes state sequences through an MLP layer, followed by Mamba layers for time-dependent feature extraction. A latent vector summarizing temporal dynamics is then utilized to estimate Q-values.

Additionally, we extend the **Prioritized Experience Replay (PER)** mechanism to store and reuse latent representations for efficient learning.

![Latent Mamba-DQN Training Flow](assets/mamba-dqn-architecture.png "Latent Mamba-DQN Training Pipeline")
---
## Experimental Results

The proposed **Latent Mamba-DQN** demonstrates superior performance in both dynamic and sparse-reward environments, compared to baseline models such as DQN, LSTM-DQN, and Transformer-DQN.

All experiments were conducted using identical replay buffer structures and consistent hyperparameter settings to ensure fair comparisons.

---

### 1. Highway-fast-v0 Environment (Dynamic Control Task)

**Comparative Performance (Smoothed Results, Averaged over 5 seeds):**

| Model            | Clipping | Avg. Smoothed Reward | Avg. Smoothed TD Loss |
|------------------|----------|----------------------|-----------------------|
| DQN              | 1.0      | 16.47                | 0.1471                |
| DQN              | 0.5      | 17.09                | 0.1368                |
| DQN              | 0.1      | 15.87                | 0.1427                |
| LSTM-DQN         | 1.0      | 12.90                | 0.0717                |
| LSTM-DQN         | 0.5      | 12.29                | 0.0796                |
| LSTM-DQN         | 0.1      | 12.21                | 0.0711                |
| Transformer-DQN  | 1.0      | 16.63                | 0.1648                |
| Transformer-DQN  | 0.5      | 15.82                | 0.1509                |
| Transformer-DQN  | 0.1      | 16.66                | 0.1467                |
| **Mamba-DQN**    | 1.0      | 17.52                | 0.0207                |
| **Mamba-DQN**    | 0.5      | **20.99**            | **0.0207**            |
| **Mamba-DQN**    | 0.1      | 20.06                | 0.0215                |

**Key Findings:**
- Mamba-DQN achieves the highest average smoothed reward across all gradient clipping settings.
- Significant reduction in TD Loss indicates enhanced learning stability.
- Mamba-DQN demonstrates faster reward improvement and consistent policy learning with lower variance after convergence.

---

### 2. LunarLander-v3 Environment (Sparse Reward Task)

**Performance Comparison (Fixed Clipping = 1.0):**

| Model           | Avg. Smoothed Reward | Avg. Smoothed TD Loss |
|-----------------|----------------------|-----------------------|
| DQN             | 27.40                | 0.0938                |
| LSTM-DQN        | 99.60                | 0.1444                |
| Transformer-DQN | 79.32                | 0.7578                |
| **Mamba-DQN**   | **224.68**           | 0.1183                |

**Observations:**
- Mamba-DQN achieves rapid reward increase during early training.
- Stable convergence at a reward level of 200 after approximately 175,000 steps, significantly outperforming baseline models.

---

*For detailed convergence curves and additional experimental results, please refer to the main publication or contact the corresponding author.*

---

## Convergence Graphs

##### Convergence Graphs

### 1. Smoothed Reward and TD Loss — Highway-fast-v0

The following graphs illustrate the learning dynamics of each model in the Highway-fast-v0 environment:

- **Top**: Smoothed Total Reward over training steps  
- **Bottom**: Smoothed TD Loss over training steps  

Mamba-DQN demonstrates superior reward improvement during early training and maintains significantly lower TD Loss, indicating enhanced learning stability and more robust policy learning.

![Highway-fast-v0 Reward Curve](assets/highway_reward_Figure.png "Smoothed reward convergence for Highway-fast-v0")

![Highway-fast-v0 TD Loss Curve](assets/highway_loss_Figure.png "Smoothed TD Loss convergence for Highway-fast-v0")

---

### 2. Smoothed Reward and TD Loss — LunarLander-v3

The following graphs present the smoothed total reward and TD Loss convergence for the LunarLander-v3 environment:

- **Top**: Smoothed Total Reward trajectory  
- **Bottom**: Smoothed TD Loss convergence  

Mamba-DQN achieves rapid reward increase, reaching stable convergence around 200 reward, while maintaining lower TD Loss than baseline models throughout training.

![LunarLander-v3 Reward Curve](assets/lunarlender_reward_Figure.png "Smoothed reward convergence for LunarLander-v3")

![LunarLander-v3 TD Loss Curve](assets/lunarlender_loss_Figure.png "Smoothed TD Loss convergence for LunarLander-v3")

---

*All results are averaged over 5 independent seeds with a smoothing coefficient of 0.9 applied to reward and loss curves.*

