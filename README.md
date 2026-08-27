# CardioFusion: Multimodal Deep Learning for HFrEF Detection

## Overview
A deep-learning system that combines electrical cardiac activity (ECG) and heart sounds (PCG) to detect heart failure with reduced ejection fraction (HFrEF). The primary research goal is evaluating whether combining these modalities improves detection performance compared to single-modality models.

## Dataset
We use a public HFrEF ECG + PCG dataset available on [Zenodo](https://zenodo.org/records/16934966).
* **Subjects:** 620
* **Recordings:** 2,480 paired 30-second recordings
* **Sampling Rates:** ECG at 500 Hz, PCG at 4,000 Hz
* **Labels:** HFrEF defined as echocardiographic LVEF ≤ 40%

## Architecture & Models
* **Baseline 1:** ECG-only 1D CNN/ResNet
* **Baseline 2:** PCG-only CNN
* **Multimodal Fusion:** Feature fusion of ECG and PCG encoders
* **Advanced Fusion:** Attention-based multimodal fusion to learn cross-modality interactions

## Main Experiments
We will evaluate and compare the models across several configurations:
* **Models compared:** ECG-only, PCG-only, simple fusion, and attention fusion.
* **Evaluation metrics:** AUROC, AUPRC, F1-score, sensitivity, specificity, balanced accuracy, and training time.
* **Ablation study:** We will conduct an ablation study to determine whether the inclusion of the second modality actually provides additional actionable information over a single-modality approach.