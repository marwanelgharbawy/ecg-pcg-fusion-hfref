"""Check whether simple heart-rate or recording-quality data predicts HFrEF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .ecg_experiments import classification_metrics, f1_midpoint_threshold
from .preprocessing import CHANNEL_ORDER


HEART_RATE_COLUMNS = ["estimated_heart_rate_bpm"]
QUALITY_COLUMNS = [
    "baseline_drift_rms_ratio",
    "high_frequency_rms_ratio",
    "full_scale_sample_fraction",
    "longest_constant_run_s",
    "detection_failed",
    "quality_flagged",
]


def make_patient_feature_table(site_features):
    data = site_features.copy()
    if data.duplicated(["Patient_ID", "Site"]).any():
        raise ValueError("Expected one row per patient and site.")
    if set(data.Site.unique()) != set(CHANNEL_ORDER):
        raise ValueError("The four expected recording sites were not found.")
    data["detection_failed"] = ~data.detection_status.eq("ok")
    data["quality_flagged"] = ~data.quality_status.eq("no_strict_flag")
    value_columns = HEART_RATE_COLUMNS + QUALITY_COLUMNS
    wide = data.pivot(index="Patient_ID", columns="Site", values=value_columns)
    wide = wide.reindex(columns=pd.MultiIndex.from_product([value_columns, CHANNEL_ORDER]))
    wide.columns = [f"{measure}__{site}" for measure, site in wide.columns]
    metadata = data.groupby("Patient_ID", sort=False)[["Fold", "LVEF", "Label"]].first()
    if not data.groupby("Patient_ID")[["Fold", "LVEF", "Label"]].nunique().eq(1).all().all():
        raise ValueError("A patient has inconsistent metadata across sites.")
    result = metadata.join(wide).reset_index()
    for measure in HEART_RATE_COLUMNS + QUALITY_COLUMNS[:4]:
        result[f"mean_{measure}"] = result[
            [f"{measure}__{site}" for site in CHANNEL_ORDER]
        ].mean(axis=1)
    result["failed_site_count"] = result[
        [f"detection_failed__{site}" for site in CHANNEL_ORDER]
    ].sum(axis=1)
    result["quality_flag_site_count"] = result[
        [f"quality_flagged__{site}" for site in CHANNEL_ORDER]
    ].sum(axis=1)
    if len(result) != 500 or result.Patient_ID.duplicated().any():
        raise ValueError("Expected 500 unique development patients.")
    return result


def _feature_columns(table, group):
    measures = {
        "heart_rate_only": HEART_RATE_COLUMNS,
        "quality_only": QUALITY_COLUMNS,
        "heart_rate_and_quality": HEART_RATE_COLUMNS + QUALITY_COLUMNS,
    }[group]
    return [f"{measure}__{site}" for measure in measures for site in CHANNEL_ORDER]


def run_simple_predictors(patient_features, seed=42):
    results = []
    predictions = []
    folds = [0, 1, 2, 3, 4]
    for scoring_fold in folds:
        tuning_fold = (scoring_fold + 1) % 5
        train_folds = [fold for fold in folds if fold not in (scoring_fold, tuning_fold)]
        train = patient_features[patient_features.Fold.isin(train_folds)]
        tune = patient_features[patient_features.Fold.eq(tuning_fold)]
        score = patient_features[patient_features.Fold.eq(scoring_fold)]
        for group in ("heart_rate_only", "quality_only", "heart_rate_and_quality"):
            columns = _feature_columns(patient_features, group)
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=1000,
                    random_state=seed,
                )),
            ])
            model.fit(train[columns], train.Label)
            tune_probabilities = model.predict_proba(tune[columns])[:, 1]
            threshold, _ = f1_midpoint_threshold(tune.Label, tune_probabilities)
            score_probabilities = model.predict_proba(score[columns])[:, 1]
            metrics = classification_metrics(score.Label, score_probabilities, threshold)
            metrics.update({
                "feature_group": group,
                "seed": seed,
                "train_folds": train_folds,
                "tuning_fold": tuning_fold,
                "scoring_fold": scoring_fold,
                "feature_count": len(columns),
            })
            results.append(metrics)
            frame = score[["Patient_ID", "Fold", "LVEF", "Label"]].copy()
            frame["Probability"] = score_probabilities
            frame["Prediction"] = score_probabilities >= threshold
            frame["Threshold"] = threshold
            frame["Feature_Group"] = group
            predictions.append(frame)
    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True)


def deep_score_associations(patient_features, prediction_files):
    rows = []
    measures = [
        "mean_estimated_heart_rate_bpm",
        "mean_baseline_drift_rms_ratio",
        "mean_high_frequency_rms_ratio",
        "mean_full_scale_sample_fraction",
        "mean_longest_constant_run_s",
        "failed_site_count",
        "quality_flag_site_count",
    ]
    for seed, path in prediction_files:
        predictions = pd.read_csv(path, dtype={"Patient_ID": str})
        joined = predictions.merge(
            patient_features[["Patient_ID"] + measures], on="Patient_ID", validate="many_to_one"
        )
        for architecture, architecture_data in joined.groupby("Architecture"):
            for label_group, group_data in [("all", architecture_data)]:
                for label in (0, 1):
                    label_data = architecture_data[architecture_data.Label.eq(label)]
                    label_group_name = "HFrEF" if label else "non-HFrEF"
                    for measure in measures:
                        valid = label_data[["Probability", measure]].dropna()
                        rho, p_value = spearmanr(valid.Probability, valid[measure])
                        rows.append({
                            "seed": seed, "architecture": architecture,
                            "patient_group": label_group_name, "measure": measure,
                            "patients": len(valid), "spearman_rho": rho,
                            "p_value_descriptive_only": p_value,
                        })
                for measure in measures:
                    valid = group_data[["Probability", measure]].dropna()
                    rho, p_value = spearmanr(valid.Probability, valid[measure])
                    rows.append({
                        "seed": seed, "architecture": architecture,
                        "patient_group": label_group, "measure": measure,
                        "patients": len(valid), "spearman_rho": rho,
                        "p_value_descriptive_only": p_value,
                    })
    return pd.DataFrame(rows)


def run_audit(project_dir):
    project_dir = Path(project_dir)
    output_dir = project_dir / "analysis_outputs/ecg_experiments/feature_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    site_features = pd.read_csv(
        project_dir / "analysis_outputs/ecg_development_patient_site_features.csv",
        dtype={"Patient_ID": str},
    )
    patient_features = make_patient_feature_table(site_features)
    results, predictions = run_simple_predictors(patient_features)
    prediction_files = [
        (seed, project_dir / f"analysis_outputs/ecg_experiments/five_fold_seed{seed}_predictions.csv")
        for seed in (42, 2026)
    ]
    if not all(path.is_file() for _, path in prediction_files):
        raise FileNotFoundError("Run both five-fold architecture comparisons first.")
    associations = deep_score_associations(patient_features, prediction_files)
    patient_features.to_csv(output_dir / "patient_features.csv", index=False)
    results.to_csv(output_dir / "simple_predictor_fold_results.csv", index=False)
    predictions.to_csv(output_dir / "simple_predictor_predictions.csv", index=False)
    associations.to_csv(output_dir / "deep_score_associations.csv", index=False)
    summary = results.groupby("feature_group")[[
        "auprc", "auroc", "recall", "specificity", "balanced_accuracy",
        "f1", "brier", "calibration_error",
    ]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output_dir / "simple_predictor_summary.csv", index=False)
    print(summary.to_string())
    return results, predictions, associations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=str(Path.cwd()))
    args = parser.parse_args()
    run_audit(Path(args.project_dir).resolve())


if __name__ == "__main__":
    main()
