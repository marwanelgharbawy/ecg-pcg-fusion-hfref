# Why did the new ECG fold-0 run score lower?

Recorded on 12 September 2026. This check uses development patients only.

## Follow-up completed

The controlled comparison proposed below has now been run. The original-loss run exactly reproduced the saved notebook 2.1 model, while the shared-loss run exactly reproduced the saved notebook 2.3 fold-0 model. This confirms that the loss calculation caused the two deterministic fold-0 training paths to differ. See `analysis_outputs/ecg_loss_comparison/seed42/README.md` for the complete result and limits.

## Main finding

The drop is real: loading the two saved models and evaluating them again on the same 100 fold-0 validation patients reproduced both results.

| Result | Original notebook 2.1 model | New notebook 2.3 fold-0 model |
|---|---:|---:|
| Training patients | 400 | 400 |
| Validation patients | 100, including 12 HFrEF | 100, including 12 HFrEF |
| AUROC | 0.956439 | 0.910985 |
| AUPRC (average precision) | 0.768608 | 0.473911 |
| Best saved epoch | 23 | 13 |
| Epochs run before stopping | 33 | 23 |
| First-epoch training loss | 0.052447 | 0.054712 |
| First-epoch validation AUROC | 0.787879 | 0.770833 |

The histories differ from the first epoch. This happens before cutoffs are chosen and before the five models are averaged. Neither the cutoff nor averaging explains why the individual new fold-0 model ranks patients less well.

## What matched

- Saved training and validation patient IDs match exactly, including their order.
- Both use seed 42 for fold 0, four ECG sites, full 30-second processed signals, and the same ECG encoder.
- Batch size 16, feature size 128, dropout 0.3, learning rate 0.001, weight decay 0.0001, focal alpha 0.88, focal gamma 2, maximum 50 epochs, and stopping patience 10 all match.
- Both use AdamW, gradient clipping at 1.0, and a learning-rate scheduler based on validation AUROC (factor 0.5, patience 3).
- Both save the model with the highest validation AUROC. The different best epochs are outcomes of different training histories, not different stopping rules.
- Git comparison against the commit that introduced the original checkpoint (`3c46954`) shows no change to the ECG encoder or data loader used here. Preprocessing changes are comments; the added separate-site class is not used by this experiment.

The audit executed only the setup portions of the current original notebook and the new training function, with no training loop. Starting model parameters and BatchNorm values, CPU/GPU random states, data-shuffling state, and the first training batch matched exactly. The new shape check, which runs in evaluation mode, did not change these starting values.

These checks establish that the current code can start identically. They do not reconstruct every detail of the historical run's environment or interactive notebook execution.

## A real code difference: focal loss

The original notebook defines its own focal loss. The new notebook imports `FocalLoss` from `src/train.py`.

- Original: calculate sigmoid probabilities, then select `p` for HFrEF and `1 - p` for non-HFrEF.
- New: calculate the same true-class probability as `exp(-binary_cross_entropy)`.

For binary labels, these formulas are mathematically equivalent. They do not use exactly the same floating-point operations.

On the same first training batch and the same model outputs:

- Both displayed exactly the same loss: `0.12097357213497162`.
- The largest difference in a parameter gradient was `2.9802322387695312e-08`.
- 449,865 of 522,433 gradient values differed numerically. The differences were tiny; this count does not measure how serious they are.
- No optimizer step was taken. No saved model was modified.

A gradient is the number used to decide how to change a weight. Small differences in these numbers can accumulate during training, producing different later scores and stopping points. This audit identified the loss calculation as a suspect. The later controlled comparison confirmed that it explains the two fold-0 training results in this setup.

The earlier statement that both runs had the same training recipe was incomplete: their intended loss and settings match, but the exact loss calculation differs.

## Other differences and missing information

| Item | Meaning for this comparison |
|---|---|
| Original cutoff comes from fold 0; new cutoff comes from all 500 held-out development predictions | Can change recall and specificity. Cannot explain the AUROC/AUPRC gap. |
| Original history saves validation loss; new history does not | Reduces the information available for examining training. Does not itself change the model. |
| New code checks one model output before training in evaluation mode | Checked directly: starting weights and random states match the original setup. |
| New seeds are 42 through 46 across the five folds | Fold 0 still uses 42. This does not explain the fold-0 difference. Across other folds, seed and patient-group effects are mixed. |
| Original hardware/library versions and initial random-state snapshots were not saved | We cannot rule out historical environment or notebook execution differences. |
| Historical processed-tensor hashes were not saved | Current paths and loading code match, and the old model still reproduces its old validation score, but byte-for-byte historical input identity cannot be verified. |

The replayed new predictions match their saved values to within `9.8e-17` (CSV rounding). Checkpoint file hashes before and after the audit match exactly.

## Decision after the controlled comparison

Keep the original checkpoint and all five new checkpoints. The five-fold method itself did not cause the individual fold-0 drop. The shared focal-loss calculation caused the training path to differ on this setup.

The project decision is to retain the original ECG and PCG baselines and preserve the five-fold runs as experiment records. If the comparison is revisited, match the original loss calculation and control the PCG crop sequence using development data. No further training is required for the current decision. This one fold and seed does not prove that either mathematically equivalent calculation is generally better.

The production loss implementation was not changed, and the controlled comparison did not use community-test data.

## Files in this folder

- `audit_fold0.py`: reproducible setup comparison, one-batch loss/gradient check, and development-only checkpoint replay. Run from the project root. It rewrites this folder's diagnostic outputs but does not train or change checkpoint files.
- `diagnostics.json`: exact setup checks, loss/gradient differences, checkpoint/source hashes, current software/hardware, and known gaps in the historical record.
- `settings_comparison.csv`: saved settings and patient-order comparisons.
- `history_comparison.csv`: original and new epoch histories side by side.
- `validation_replay_summary.csv`: replayed validation metrics and best-epoch information.
- `validation_replay_predictions.csv`: 200 rows, one prediction per validation patient for each of the two saved models.

Existing five-fold results and notebook outputs were preserved. This audit does not use community-test data or select a new model.
