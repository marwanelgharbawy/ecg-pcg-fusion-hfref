# ECG experiment record

Sections 1–4 use only the 500 development patients. The community test set was not used to choose experiments, epochs, models, or probability cutoffs. Section 5 records the later test evaluation requested after the model and cutoff rules had been fixed.

## 1. Separate-site model

**Question:** Does it help to process APEX, LLSB, LUSB, and RUSB separately before making one patient prediction?

**Changed:** The same one-channel encoder processed each site. The four summaries were joined in the fixed site order.

**Kept the same:** Processed ECG tensors, patient folds, focal loss, optimizer, batch size, training limit, early stopping, and patient-level scoring.

The old fold-0 checkpoint replayed at 0.769 average precision and 0.956 AUROC. A new four-channel run reached 0.490 and 0.905, while the new separate-site run reached 0.620 and 0.942. The old training score was not reproduced exactly, so the newly trained pair is the fair comparison.

In the first five-fold test, every patient was scored once by a model that did not use that patient for training or tuning. This was repeated with two starting seeds. These results are kept as a **pre-correction record** because the model check described in section 4 changed internal training values before the first training batch:

| Seed | Model | Mean average precision | Mean AUROC | Recall | Specificity | Balanced accuracy |
|---:|---|---:|---:|---:|---:|---:|
| 42 | Four-channel | 0.524 | 0.873 | 0.667 | 0.880 | 0.773 |
| 42 | Separate-site | 0.604 | 0.898 | 0.700 | 0.859 | 0.780 |
| 2026 | Four-channel | 0.539 | 0.889 | 0.783 | 0.884 | 0.834 |
| 2026 | Separate-site | 0.569 | 0.905 | 0.767 | 0.850 | 0.808 |

Across the ten fold-and-seed comparisons, average precision improved in 9. The average gain was about 0.055. Specificity fell by about 0.027, and balanced accuracy did not improve overall. The patient-resampling ranges for the average-precision gain included zero for both seeds, so the gain remains uncertain.

**Old decision:** Keep the separate-site code as an experiment, but do not replace the current model yet. Section 4 replaces this decision after corrected training.

## 2. Heart-rate and recording-quality audit

**Question:** Can simple heart-rate or recording-quality measurements predict the label, and are they related to neural-network scores?

A fixed logistic model used the same training, tuning, and scoring fold roles:

| Inputs | Mean average precision | Mean AUROC |
|---|---:|---:|
| Heart rate only | 0.206 | 0.640 |
| Recording quality only | 0.383 | 0.696 |
| Heart rate and quality | 0.404 | 0.728 |

The HFrEF rate is 0.12, so recording-quality information separates the labels more than expected by chance. The strongest checked link between a deep-model score and one audit measure had an absolute Spearman rank correlation of about 0.32. This is a moderate association. It does not prove the neural network uses noise.

**Decision:** Treat recording conditions as a possible shortcut. Investigate available device, visit, operator, and acquisition information before trusting a small model gain or choosing noise augmentation.

## 3. Training crop comparison

**Question:** Does showing a different 15-second section during training help the current four-channel model?

All tuning and scoring used the full 30 seconds.

| Training input | Mean average precision | Recall | Specificity | Balanced accuracy |
|---|---:|---:|---:|---:|
| Full 30 seconds | 0.524 | 0.667 | 0.880 | 0.773 |
| Fixed middle 15 seconds | 0.553 | 0.450 | 0.927 | 0.689 |
| Random 15 seconds | 0.476 | 0.617 | 0.877 | 0.747 |

**Decision:** Reject random 15-second cropping for now. The fixed crop had slightly higher average precision but missed many more HFrEF patients, so it is also not a safe replacement.

## Deferred ideas

- **Mild synthetic noise:** Deferred because recording quality already carries label information and the development recordings do not define one clearly safe noise type and strength after preprocessing.
- **Loss or balanced sampling:** Deferred because focal loss already gives HFrEF patients more weight, and the recall changes were not consistent across folds and seeds.
- **Strong amplitude changes, time stretching, beat warping, and site permutation:** Avoid for now because they may create unrealistic signals or damage useful information.

## 4. Corrected training and fair cutoff comparison

The code checked the model's output shape using tuning patients while the model was in training mode. This changed BatchNorm running values and used random dropout numbers before training started. The check now uses evaluation mode without gradients and restores the earlier mode. Both models were trained again across all five folds with seeds 42 and 2026.

We also used a second cutoff rule. It requires at least 90% specificity on the tuning fold, then allows as many positive predictions as possible. This makes recall comparisons easier to understand because both models aim for the same tuning specificity. Scoring patients do not choose this cutoff.

| Seed | Model | Recall | Specificity | Precision | F1 | Balanced accuracy | Average precision | AUROC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | Four-channel | 0.633 | 0.891 | 0.461 | 0.509 | 0.762 | 0.534 | 0.876 |
| 42 | Separate-site | 0.717 | 0.902 | 0.510 | 0.576 | 0.809 | 0.621 | 0.916 |
| 2026 | Four-channel | 0.533 | 0.898 | 0.412 | 0.457 | 0.716 | 0.482 | 0.880 |
| 2026 | Separate-site | 0.633 | 0.893 | 0.454 | 0.513 | 0.763 | 0.557 | 0.886 |

Separate-site processing improved recall with both seeds by 0.083 and 0.100. Specificity changed by +0.011 and -0.005. Average precision also improved with both seeds. Separate-site recall improved in five of the ten scoring fold runs and stayed equal in the other five.

Patient resampling still shows uncertainty around the recall improvement because there are only 60 HFrEF patients. The quality-group check did not show that the gain came from one clear signal-quality group. Some quality groups contained very few HFrEF patients, and only three patients had a failed automatic beat check.

**Development decision:** Use separate-site processing as the model to develop next. Keep the old four-channel checkpoint as the historical reference until a final model-training plan is agreed. This decision was made before opening the community test set; section 5 records the later test result.

## 5. Community test comparison

The separate-site checkpoint was selected before reading test labels. It was the checkpoint with the highest average precision on its held-out development scoring fold: seed 42, scoring fold 1, average precision 0.759.

The primary comparison gave both models the same cutoff rule. Each cutoff was chosen from that model's development tuning patients to reach at least 90% specificity while allowing as many positive predictions as possible. No test patient chose either cutoff.

The community test set contains 120 patients: 5 HFrEF and 115 non-HFrEF.

| Model | Recall | Specificity | Precision | F1 | Balanced accuracy | Average precision | AUROC | False warnings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Notebook 2 four-channel | 1.000 | 0.965 | 0.556 | 0.714 | 0.983 | 0.967 | 0.998 | 4 |
| Best separate-site | 1.000 | 0.878 | 0.263 | 0.417 | 0.939 | 0.900 | 0.991 | 14 |

Both models found all five HFrEF patients. The separate-site model created ten more false warnings and did not improve average precision or AUROC. On this test set, the notebook 2 four-channel model is the better model to keep.

Only five test patients have HFrEF, so one patient would change recall by 20 percentage points. This test result should be reported as the final held-out check and should not be used to tune another version of the model.

## Files

- `fold0_architecture_comparison.csv`: historical replay and the first controlled fold-0 comparison.
- `five_fold_seed*_results.csv`: one result row for each model and scoring fold.
- `five_fold_seed*_predictions.csv`: one score per development patient for each model.
- `five_fold_seed*_paired_uncertainty.csv`: separate-site minus four-channel differences with patient-resampling ranges.
- `feature_audit/`: patient features, simple-predictor results, predictions, and associations with deep-model scores.
- `crop_comparison_seed42_*`: full, fixed-crop, and random-crop results and patient scores.
- `five_fold_corrected_seed*_*`: corrected results, predictions, and patient-resampling ranges for both seeds.
- `cutoff_reanalysis_pre_correction/`: the old saved predictions checked with both cutoff rules.
- `cutoff_reanalysis_corrected/`: corrected cutoff comparisons and signal-quality group checks.
- `community_test/`: the fixed model choice, test probabilities, and test comparison results.
- `resource_check.csv`: parameter counts and peak GPU memory from one batch. The four-channel model used about 249 MiB; the separate-site model used about 890 MiB because four site encodings are held during the backward pass.

Detailed checkpoints, histories, tuning predictions, scoring predictions, settings, and runtime are under `models/ecg_experiments/`. None of these experiments establish clinical usefulness.
