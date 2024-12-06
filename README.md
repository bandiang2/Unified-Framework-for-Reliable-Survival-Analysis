# Unified-Framework-for-Reliable-Survival-Analysis

Survival analysis (time-to-event prediction) focuses on predicting the time until an event of interest, such as disease relapse or death. It is a critical tool for clinical decision-making. An effective survival model must address model understandability, quantifying uncertainty, and predictive resilience for practical adoption and trustworthiness in clinical settings. While progress has been made in these areas, existing methods often address these aspects separately, leading to limited performance and practical challenges. In this paper, we propose a principled framework that addresses these limitations in a unified approach. Our method incorporates a stochastic, biologically inspired sparsity mechanism for feature selection and applies a Bayesian treatment of model weights, enabling double stochasticity through two distinct sampling processes. These mechanisms enhance model understandability and enable robust uncertainty quantification. Furthermore, we introduce an information-bound regularization inspired by information theory, which controls information flow by promoting the learning of essential information while filtering out irrelevant and redundant data. This regularization improves generalization and enhances predictive resilience to noise and adversarial perturbations. Through extensive experiments on five real-world survival datasets, our approach demonstrates competitive performance in both discrimination and calibration compared to state-of-the-art deep survival models. On a synthetic dataset, our model effectively identifies relevant features, showcasing its ability to support reliable feature selection. Additionally, the robustness of the proposed approach to adversarial attacks highlights its predictive resilience, further reinforcing its trustworthiness for survival analysis applications.

## Requirements
`pip install -r requirements.txt`

## Run
- Download pre-processed datasets from this [link](https://drive.google.com/drive/folders/13i1hW6PT7W698ZCxjFvARznxYk8RtCA0?usp=sharing).
- ` python3 run_model.py`

