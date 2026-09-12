# Community test comparison

The new checkpoint was selected before reading test labels. It is the separate-site model from seed 42 and scoring fold 1, which had the highest average precision among the corrected separate-site checkpoints on a held-out development scoring fold.

The main comparison uses the same cutoff rule for both models: reach at least 90% specificity on that model's development tuning patients, then allow as many positive predictions as possible. Test patients did not choose the models or cutoffs.

The test set has 120 patients: 5 HFrEF and 115 non-HFrEF.

| Model | Development cutoff | Recall | Specificity | Precision | F1 | Balanced accuracy | Average precision | AUROC | TP | FN | TN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Notebook 2.1 four-channel | 0.4982 | 1.000 | 0.965 | 0.556 | 0.714 | 0.983 | 0.967 | 0.998 | 5 | 0 | 111 | 4 |
| Best separate-site | 0.1212 | 1.000 | 0.878 | 0.263 | 0.417 | 0.939 | 0.900 | 0.991 | 5 | 0 | 101 | 14 |

Both models found all five HFrEF patients. The separate-site model produced ten more false warnings. The notebook 2.1 four-channel model performed better on this test set.

Because there are only five HFrEF test patients, one patient changes recall by 20 percentage points. Do not use these results to tune another model.

Files:

- `selection_and_setup.json`: the model-selection rule, selected fold and seed, test counts, and tuning specificity.
- `community_test_results.csv`: all reported measures for both cutoff rules, plus notebook 2.1's saved cutoff.
- `community_test_predictions.csv`: one probability and prediction per patient, model, and shared cutoff rule.
