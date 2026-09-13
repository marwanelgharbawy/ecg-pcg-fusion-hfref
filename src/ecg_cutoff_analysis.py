"""Compare ECG probability cutoffs and recording-quality groups."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .ecg_experiments import (
    classification_metrics,
    f1_midpoint_threshold,
    paired_bootstrap_differences,
)


TARGET_SPECIFICITY = 0.90
SEEDS = (42, 2026)
ARCHITECTURES = ("joint_4ch", "shared_site")
QUALITY_MEASURES = (
    "mean_baseline_drift_rms_ratio",
    "mean_high_frequency_rms_ratio",
)


def _midpoint_for_predictions(probabilities, predicted):
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = np.asarray(predicted, dtype=bool)
    positive_scores = probabilities[predicted]
    negative_scores = probabilities[~predicted]
    if len(positive_scores) and len(negative_scores):
        threshold = float((positive_scores.min() + negative_scores.max()) / 2)
    elif len(positive_scores):
        threshold = float(np.nextafter(positive_scores.min(), -np.inf))
    else:
        threshold = float(np.nextafter(probabilities.max(), np.inf))
    if not np.array_equal(predicted, probabilities >= threshold):
        raise AssertionError("Midpoint changed the selected classifications.")
    return threshold


def specificity_target_threshold(labels, probabilities, target=TARGET_SPECIFICITY):
    """Allow the most positive predictions while meeting tuning specificity."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Cutoff selection requires both classes.")
    candidates = np.r_[
        np.nextafter(probabilities.max(), np.inf),
        np.unique(probabilities),
    ]
    choices = []
    for threshold in candidates:
        predicted = probabilities >= threshold
        negatives = labels == 0
        specificity = float((~predicted[negatives]).mean())
        if specificity + 1e-12 >= target:
            choices.append((int(predicted.sum()), float(threshold), predicted, specificity))
    if not choices:
        raise RuntimeError("No cutoff met the specificity target.")
    # More positive predictions are preferred. The higher cutoff is a stable
    # tie-breaker, although unique score cutoffs normally give unique predictions.
    _, _, predicted, _ = max(choices, key=lambda item: (item[0], item[1]))
    threshold = _midpoint_for_predictions(probabilities, predicted)
    metrics = classification_metrics(labels, probabilities, threshold)
    if metrics["specificity"] + 1e-12 < target:
        raise AssertionError("The midpoint failed the tuning specificity target.")
    return threshold, metrics


def _validate_saved_frames(metadata, tuning, scoring, scoring_fold):
    tuning_fold = (scoring_fold + 1) % 5
    for name, frame, expected_fold in (
        ("tuning", tuning, tuning_fold),
        ("scoring", scoring, scoring_fold),
    ):
        required = {
            "Patient_ID", "Fold", "Label", "LVEF", "Probability",
            "Prediction", "Threshold", "Architecture",
        }
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} predictions are missing required columns.")
        if len(frame) != 100 or frame.Patient_ID.duplicated().any():
            raise ValueError(f"Expected 100 unique {name} patients.")
        if set(frame.Fold.unique()) != {expected_fold}:
            raise ValueError(f"Unexpected {name} fold.")
        expected = metadata[metadata.Fold.eq(expected_fold)][
            ["Patient_ID", "Label", "LVEF"]
        ].sort_values("Patient_ID").reset_index(drop=True)
        actual = frame[["Patient_ID", "Label", "LVEF"]].sort_values(
            "Patient_ID"
        ).reset_index(drop=True)
        if not expected.Patient_ID.equals(actual.Patient_ID):
            raise ValueError(f"{name} patient IDs do not match metadata.")
        if not np.array_equal(expected.Label, actual.Label):
            raise ValueError(f"{name} labels do not match metadata.")
        if not np.allclose(expected.LVEF, actual.LVEF):
            raise ValueError(f"{name} LVEF values do not match metadata.")
        if not np.isfinite(frame.Probability).all():
            raise ValueError(f"{name} probabilities contain invalid values.")
    if set(tuning.Patient_ID) & set(scoring.Patient_ID):
        raise ValueError("Tuning and scoring patients overlap.")


def _rule_results(tuning, scoring, rule):
    labels = tuning.Label.to_numpy(dtype=int)
    probabilities = tuning.Probability.to_numpy(dtype=float)
    if rule == "maximum_tuning_f1":
        threshold, _ = f1_midpoint_threshold(labels, probabilities)
        tuning_metrics = classification_metrics(labels, probabilities, threshold)
    elif rule == "target_90pct_tuning_specificity":
        threshold, tuning_metrics = specificity_target_threshold(labels, probabilities)
    else:
        raise ValueError(f"Unknown cutoff rule: {rule}")
    scoring_metrics = classification_metrics(
        scoring.Label, scoring.Probability, threshold
    )
    return threshold, tuning_metrics, scoring_metrics


def reanalyze_saved_predictions(project_dir, source):
    """Apply two tuning-only cutoff rules to existing patient predictions."""
    project_dir = Path(project_dir)
    if source not in ("pre_correction", "corrected"):
        raise ValueError("source must be pre_correction or corrected.")
    input_stem = "five_fold" if source == "pre_correction" else "five_fold_corrected"
    output_dir = (
        project_dir / "analysis_outputs/ecg_experiments"
        / f"cutoff_reanalysis_{source}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Analysis directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(
        project_dir / "data/processed/patient_4ch/development/processed_metadata.csv",
        dtype={"Patient_ID": str},
    )
    if len(metadata) != 500 or metadata.Patient_ID.duplicated().any():
        raise ValueError("Expected 500 unique development patients.")

    result_rows = []
    prediction_frames = []
    rules = ("maximum_tuning_f1", "target_90pct_tuning_specificity")
    for seed in SEEDS:
        for scoring_fold in range(5):
            for architecture in ARCHITECTURES:
                run_dir = (
                    project_dir / "models/ecg_experiments"
                    / f"{input_stem}_seed{seed}" / f"score_fold_{scoring_fold}"
                    / architecture
                )
                tuning = pd.read_csv(
                    run_dir / "tuning_predictions.csv", dtype={"Patient_ID": str}
                )
                scoring = pd.read_csv(
                    run_dir / "scoring_predictions.csv", dtype={"Patient_ID": str}
                )
                _validate_saved_frames(metadata, tuning, scoring, scoring_fold)
                old_predictions = scoring.Probability.ge(scoring.Threshold).to_numpy()
                if not np.array_equal(old_predictions, scoring.Prediction.to_numpy(bool)):
                    raise ValueError("Saved predictions do not match their saved cutoff.")
                run_rank_metrics = None
                for rule in rules:
                    threshold, tune_metrics, score_metrics = _rule_results(
                        tuning, scoring, rule
                    )
                    ranking = {
                        key: score_metrics[key] for key in ("auprc", "auroc", "brier")
                    }
                    if run_rank_metrics is None:
                        run_rank_metrics = ranking
                    elif any(
                        not np.isclose(ranking[key], run_rank_metrics[key])
                        for key in ranking
                    ):
                        raise AssertionError("Cutoff changed a cutoff-free metric.")
                    result_rows.append({
                        "source": source,
                        "seed": seed,
                        "scoring_fold": scoring_fold,
                        "tuning_fold": (scoring_fold + 1) % 5,
                        "architecture": architecture,
                        "cutoff_rule": rule,
                        "cutoff": threshold,
                        "tuning_specificity": tune_metrics["specificity"],
                        "tuning_recall": tune_metrics["recall"],
                        **{f"scoring_{key}": value for key, value in score_metrics.items()},
                    })
                    frame = scoring[["Patient_ID", "Fold", "Label", "LVEF", "Probability"]].copy()
                    frame["Prediction"] = frame.Probability.ge(threshold)
                    frame["Threshold"] = threshold
                    frame["Architecture"] = architecture
                    frame["Seed"] = seed
                    frame["Cutoff_Rule"] = rule
                    frame["Source"] = source
                    prediction_frames.append(frame)

    results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    expected_rows = len(SEEDS) * 5 * len(ARCHITECTURES) * len(rules)
    if len(results) != expected_rows:
        raise AssertionError("Unexpected number of cutoff result rows.")
    key = ["Seed", "Cutoff_Rule", "Architecture", "Patient_ID"]
    if predictions.duplicated(key).any():
        raise ValueError("A patient was scored more than once in a comparison.")
    if not predictions.groupby(["Seed", "Cutoff_Rule", "Architecture"]).size().eq(500).all():
        raise ValueError("Every comparison must contain 500 scored patients.")

    results.to_csv(output_dir / "cutoff_results_by_fold.csv", index=False)
    predictions.to_csv(output_dir / "cutoff_predictions.csv", index=False)
    summary_columns = [
        "scoring_recall", "scoring_specificity", "scoring_precision", "scoring_f1",
        "scoring_balanced_accuracy", "scoring_auprc", "scoring_auroc",
        "scoring_brier", "tuning_specificity", "tuning_recall",
    ]
    summary = results.groupby(
        ["seed", "cutoff_rule", "architecture"]
    )[summary_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output_dir / "cutoff_summary.csv", index=False)

    uncertainty_frames = []
    for seed in SEEDS:
        for rule in rules:
            part = predictions[
                predictions.Seed.eq(seed) & predictions.Cutoff_Rule.eq(rule)
            ]
            uncertainty = paired_bootstrap_differences(
                part, n_bootstrap=2000, seed=seed + (0 if rule == rules[0] else 100000)
            )
            uncertainty.insert(0, "cutoff_rule", rule)
            uncertainty.insert(0, "seed", seed)
            uncertainty_frames.append(uncertainty)
    uncertainty = pd.concat(uncertainty_frames, ignore_index=True)
    uncertainty.to_csv(output_dir / "paired_architecture_uncertainty.csv", index=False)
    return results, predictions, uncertainty, output_dir


def _quality_boundaries(patient_features, scoring_fold):
    tuning_fold = (scoring_fold + 1) % 5
    training = patient_features[
        ~patient_features.Fold.isin([scoring_fold, tuning_fold])
    ]
    rows = []
    for measure in QUALITY_MEASURES:
        low, high = training[measure].quantile([1 / 3, 2 / 3]).to_numpy()
        rows.append({
            "scoring_fold": scoring_fold,
            "tuning_fold": tuning_fold,
            "measure": measure,
            "low_boundary": low,
            "high_boundary": high,
            "boundary_patients": len(training),
        })
    return rows


def _group_name(value, low, high):
    if pd.isna(value):
        return "missing"
    if value <= low:
        return "low"
    if value <= high:
        return "middle"
    return "high"


def _group_metrics(frame):
    labels = frame.Label.to_numpy(dtype=int)
    predicted = frame.Prediction.to_numpy(dtype=bool)
    positives = labels == 1
    negatives = labels == 0
    tp = int((predicted & positives).sum())
    fn = int((~predicted & positives).sum())
    tn = int((~predicted & negatives).sum())
    fp = int((predicted & negatives).sum())
    return {
        "patients": len(frame),
        "hfref_patients": int(positives.sum()),
        "non_hfref_patients": int(negatives.sum()),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


def analyze_quality_groups(project_dir, source):
    project_dir = Path(project_dir)
    output_dir = (
        project_dir / "analysis_outputs/ecg_experiments"
        / f"cutoff_reanalysis_{source}"
    )
    predictions = pd.read_csv(
        output_dir / "cutoff_predictions.csv", dtype={"Patient_ID": str}
    )
    features = pd.read_csv(
        project_dir / "analysis_outputs/ecg_experiments/feature_audit/patient_features.csv",
        dtype={"Patient_ID": str},
    )
    boundaries = pd.DataFrame([
        row for fold in range(5) for row in _quality_boundaries(features, fold)
    ])
    boundary_lookup = boundaries.set_index(["scoring_fold", "measure"])
    merged = predictions.merge(
        features[["Patient_ID", *QUALITY_MEASURES, "failed_site_count"]],
        on="Patient_ID", validate="many_to_one",
    )
    group_frames = []
    for measure in QUALITY_MEASURES:
        part = merged.copy()
        part["Quality_Measure"] = measure
        part["Quality_Group"] = [
            _group_name(
                value,
                boundary_lookup.loc[(fold, measure), "low_boundary"],
                boundary_lookup.loc[(fold, measure), "high_boundary"],
            )
            for value, fold in zip(part[measure], part.Fold)
        ]
        group_frames.append(part)
    failure = merged.copy()
    failure["Quality_Measure"] = "beat_detection_check"
    failure["Quality_Group"] = np.where(
        failure.failed_site_count.eq(0), "all_sites_passed", "one_or_more_failed"
    )
    group_frames.append(failure)
    grouped_predictions = pd.concat(group_frames, ignore_index=True)

    keys = [
        "Seed", "Cutoff_Rule", "Architecture", "Fold",
        "Quality_Measure", "Quality_Group",
    ]
    fold_rows = []
    for values, frame in grouped_predictions.groupby(keys, sort=True):
        fold_rows.append(dict(zip(keys, values)) | _group_metrics(frame))
    fold_results = pd.DataFrame(fold_rows)

    summary_keys = [
        "Seed", "Cutoff_Rule", "Architecture", "Quality_Measure", "Quality_Group",
    ]
    summary_rows = []
    for values, frame in grouped_predictions.groupby(summary_keys, sort=True):
        summary_rows.append(dict(zip(summary_keys, values)) | _group_metrics(frame))
    summary = pd.DataFrame(summary_rows)
    boundaries.to_csv(output_dir / "quality_group_boundaries.csv", index=False)
    fold_results.to_csv(output_dir / "quality_group_results_by_fold.csv", index=False)
    summary.to_csv(output_dir / "quality_group_summary.csv", index=False)
    return boundaries, fold_results, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["pre_correction", "corrected"])
    parser.add_argument("--project-dir", default=str(Path.cwd()))
    args = parser.parse_args()
    project_dir = Path(args.project_dir).resolve()
    results, _, uncertainty, output_dir = reanalyze_saved_predictions(
        project_dir, args.source
    )
    _, _, quality = analyze_quality_groups(project_dir, args.source)
    print(results.groupby(["cutoff_rule", "architecture"])[[
        "scoring_recall", "scoring_specificity", "scoring_precision",
        "scoring_f1", "scoring_balanced_accuracy", "scoring_auprc", "scoring_auroc",
    ]].mean().to_string())
    print("\nPaired differences are saved to:", output_dir)
    print("Quality-group rows:", len(quality), "Uncertainty rows:", len(uncertainty))


if __name__ == "__main__":
    main()
