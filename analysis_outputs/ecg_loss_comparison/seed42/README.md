# ECG fold-0 focal-loss comparison

This experiment used development data only. It trained two models on folds 1–4 and validated them on fold 0. Both runs used the same model, starting weights, seed 42, patient order, shuffle state, optimizer, scheduler, and stopping rule. Only the focal-loss calculation changed.

## Results

| Model | AUROC | AUPRC | Best epoch | Epochs run | HFrEF found at its F1 cutoff | False alarms |
|---|---:|---:|---:|---:|---:|---:|
| Saved original notebook 2.1 | 0.956439 | 0.768608 | 23 | 33 | Not recalculated here | Not recalculated here |
| New run with original notebook loss | 0.956439 | 0.768608 | 23 | 33 | 11 of 12 | 8 |
| Saved notebook 2.3 fold 0 | 0.910985 | 0.473911 | 13 | 23 | Not recalculated here | Not recalculated here |
| New run with shared source loss | 0.910985 | 0.473911 | 13 | 23 | 9 of 12 | 7 |

The new run using the original notebook loss exactly reproduced the saved original model:

- Every saved model tensor is identical.
- All common training-history values are identical.
- All 100 validation probabilities are identical.

The new run using the shared source loss exactly reproduced the saved notebook 2.3 fold-0 model in the same three checks.

This confirms that the focal-loss calculation caused the two training paths to separate in this deterministic fold-0 setup. The formulas are mathematically equivalent, but they calculate gradients through different floating-point operations. Neural-network training can amplify very small numerical differences over many weight updates.

## What this establishes

- The original fold-0 checkpoint can be reproduced on the current machine.
- The shared loss explains the lower fold-0 result in the current five-fold experiment.
- Cutoff selection and five-model averaging did not cause the individual fold-0 AUROC/AUPRC difference.

## What it does not establish

- It does not show that the original loss calculation is generally better across all folds or seeds.
- It does not show how either calculation performs on PCG.
- It does not establish a clinical performance difference.

## Decision

Keep the original ECG and PCG models as the project baselines. Preserve the five-fold runs and this loss comparison as experiment records. The PCG five-fold notebook has since been run with the shared loss; its results and limitations are recorded in notebook 3.2. The original PCG checkpoint is missing locally and must be recovered before using that model locally or for fusion.

The intended five-fold experiment was supposed to change fold use while keeping the old training method. These runs did not exactly preserve it. If the comparison is revisited, match each original notebook's exact loss calculation and, for PCG, control the random training-crop sequence. This is an optional development-only follow-up, not a required rerun. The ECG loss finding does not establish the cause of the PCG result.

The production `FocalLoss` in `src/train.py` was not changed. This controlled comparison did not use community-test data.

## Saved files

- `comparison_summary.csv`: the two controlled results and two saved-checkpoint results.
- `setup_checks.json`: confirms matching starting setup and that no community-test data was loaded.
- `reproduction_checks.json`: confirms exact model-state, history, and validation-probability reproduction.
- `models/ecg_loss_comparison/seed42/original_notebook_loss/`: original-loss model, history, settings, and validation predictions.
- `models/ecg_loss_comparison/seed42/shared_src_loss/`: shared-loss model, history, settings, and validation predictions.
- `notebooks/2_4_ecg_loss_comparison.ipynb`: executed notebook with tables and training curves.

Existing checkpoints and five-fold results were preserved.
