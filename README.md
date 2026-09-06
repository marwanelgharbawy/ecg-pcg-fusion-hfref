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

## ECG Model
The ECG model now looks at all 4 channels from the same patient together, instead of treating each recording as a separate input.

* **Input:** APEX, LLSB, LUSB, and RUSB, always in this order. Each channel is 30 seconds at 500 Hz, so one patient has an input shape of `(4, 15000)`.
* **Preprocessing:** We apply a 0.5–40 Hz Butterworth filter and Z-score normalization separately to each channel. We keep the full recording without segmentation and save one tensor and one metadata row per patient.
* **Convolution Layers:** The first convolution takes the 4 channels and produces 32 feature maps, followed by batch normalization, ReLU, and max pooling. Then 4 residual blocks use 32, 64, 128, and 128 feature maps. Each block has two convolutions with kernel size 7 and a skip connection, with dropout of 0.1 between the convolutions.
* **Classifier:** Global average and max pooling are combined into 256 features. A fully connected layer reduces them to 128 features, followed by ReLU, dropout of 0.3, and one output per patient. Sigmoid converts this output into a score for HFrEF.

Run `notebooks/1_preprocessing.ipynb` first to prepare both the development and test sets, then `notebooks/2_ecg_baseline.ipynb` to train and evaluate the ECG model. All 4 channels stay with their patient when splitting the data. The classification threshold is chosen on validation patients and kept the same for the test set.
