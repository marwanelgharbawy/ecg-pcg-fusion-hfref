"""Audit the two ECG fold-0 runs without training or reading community test data.

Run from the project root with:
python analysis_outputs/five_fold_cross_validation/ecg_fold0_audit/audit_fold0.py
"""

import ast
import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src import five_fold_validation as five
from src.ecg_model import ECG_Encoder
from src.train import FocalLoss

OUT = Path(__file__).resolve().parent
OLD = ROOT / "models/best_ecg_4ch_fold0.pt"
NEW = ROOT / "models/five_fold_cross_validation/ecg_seed42/fold_0/best_model.pt"


def digest_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(model, loader):
    return {
        "state": {key: digest_tensor(value) for key, value in model.state_dict().items()},
        "cpu_rng": digest_tensor(torch.get_rng_state()),
        "cuda_rng": [digest_tensor(value) for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
        "loader_rng": digest_tensor(loader.generator.get_state()),
    }


def main():
    hashes_before = {str(path.relative_to(ROOT)): digest_file(path) for path in (OLD, NEW)}
    notebook_path = ROOT / "notebooks/2_1_ecg_baseline.ipynb"
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    legacy = {}
    # Execute only imports/seed/metadata, loader/model setup, and the loss class.
    # The notebook's training, saving, and test cells are never executed.
    for index in (1, 3, 6):
        exec(compile("".join(nb["cells"][index]["source"]), f"baseline_cell_{index}", "exec"), legacy)
    old_setup = snapshot(legacy["ecg_model"], legacy["train_loader"])
    old_first_batch = next(iter(legacy["train_loader"]))

    # Execute the actual new function's setup, ending before any output directory
    # is created or any training loop starts. Return its local objects for checks.
    tree = ast.parse(inspect.getsource(five._train_one_fold))
    function = tree.body[0]
    cutoff = next(i for i, node in enumerate(function.body)
                  if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "checkpoint_dir" for t in node.targets))
    function.body = function.body[:cutoff] + [ast.Return(value=ast.Call(
        func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]))]
    ast.fix_missing_locations(tree)
    namespace = dict(vars(five))
    exec(compile(tree, "new_setup_only", "exec"), namespace)
    new = namespace["_train_one_fold"](
        ROOT, "ecg", 42, 0, 16, 0.001, 50, 10, OUT / "unused"
    )
    new_setup = snapshot(new["model"], new["train_loader"])
    new_first_batch = next(iter(new["train_loader"]))
    startup = {
        "initial_model_state_equal": old_setup["state"] == new_setup["state"],
        "cpu_random_state_equal": old_setup["cpu_rng"] == new_setup["cpu_rng"],
        "gpu_random_state_equal": old_setup["cuda_rng"] == new_setup["cuda_rng"],
        "shuffle_generator_state_equal": old_setup["loader_rng"] == new_setup["loader_rng"],
        "first_batch_signals_equal": torch.equal(old_first_batch[0], new_first_batch[0]),
        "first_batch_labels_equal": torch.equal(old_first_batch[1], new_first_batch[1]),
        "training_patient_order_equal": legacy["train_df"].Patient_ID.tolist() == new["train_df"].Patient_ID.tolist(),
        "validation_patient_order_equal": legacy["val_df"].Patient_ID.tolist() == new["validation_df"].Patient_ID.tolist(),
        "shape_check_preserves_state_and_rng": old_setup == new_setup,
    }

    # One forward pass and two gradient calculations on the same graph.
    # No optimizer step, weight update, or model save is performed.
    model = new["model"]
    values, labels = new_first_batch
    logits = model(values.to(new["device"]))
    labels = labels.to(new["device"]).reshape(-1, 1)
    old_loss = legacy["FocalLoss"](alpha=0.88, gamma=2.0)(logits, labels)
    new_loss = FocalLoss(alpha=0.88, gamma=2.0)(logits, labels)
    parameters = tuple(model.parameters())
    old_grad = torch.autograd.grad(old_loss, parameters, retain_graph=True)
    new_grad = torch.autograd.grad(new_loss, parameters)
    max_grad_difference = max((a-b).abs().max().item() for a, b in zip(old_grad, new_grad))
    loss_check = {
        "legacy_loss": old_loss.item(),
        "shared_loss": new_loss.item(),
        "absolute_loss_difference": abs(old_loss.item()-new_loss.item()),
        "maximum_parameter_gradient_difference": max_grad_difference,
        "gradient_elements_different": sum(int((a != b).sum()) for a, b in zip(old_grad, new_grad)),
        "gradient_elements_total": sum(a.numel() for a in old_grad),
        "optimizer_steps": 0,
    }
    del logits, old_loss, new_loss, old_grad, new_grad

    old_settings = json.loads((ROOT / "models/best_ecg_4ch_fold0.json").read_text())
    new_settings = json.loads((NEW.parent / "settings.json").read_text())
    comparison = []
    for key in ("model_config", "batch_size", "learning_rate", "weight_decay", "focal_alpha", "focal_gamma", "max_epochs", "patience", "validation_fold"):
        comparison.append({"setting": key, "old": str(old_settings[key]), "new": str(new_settings[key]), "equal": old_settings[key] == new_settings[key]})
    for name, old_value, new_value in (
        ("seed", old_settings["seed"], new_settings["fold_seed"]),
        ("training_patient_order", old_settings["train_patient_ids"], new_settings["training_patient_ids"]),
        ("validation_patient_order", old_settings["validation_patient_ids"], new_settings["validation_patient_ids"]),
    ):
        comparison.append({"setting": name, "old": str(old_value) if name == "seed" else f"{len(old_value)} patients", "new": str(new_value) if name == "seed" else f"{len(new_value)} patients", "equal": old_value == new_value})
    pd.DataFrame(comparison).to_csv(OUT / "settings_comparison.csv", index=False)

    old_history = pd.read_csv(ROOT / "models/best_ecg_4ch_fold0.history.csv")
    new_history = pd.read_csv(NEW.parent / "history.csv").rename(columns={"learning_rate": "lr"})
    old_history.merge(new_history, on="epoch", how="outer", suffixes=("_old", "_new")).to_csv(
        OUT / "history_comparison.csv", index=False)

    metadata, data_dir = five._load_metadata(ROOT, "development")
    validation = metadata[metadata.Fold.eq(0)].reset_index(drop=True)
    loader = torch.utils.data.DataLoader(
        five.ProcessedCardioDataset(validation, data_dir, modality="ecg"),
        batch_size=16, shuffle=False, num_workers=0,
    )
    replay_rows = []
    replay_predictions = []
    saved_new_predictions = pd.read_csv(NEW.parent / "validation_predictions.csv", dtype={"Patient_ID": str})
    for name, path, settings in (("original", OLD, old_settings), ("new_fold0", NEW, new_settings)):
        replay_model = ECG_Encoder(**settings["model_config"]).to(new["device"])
        replay_model.load_state_dict(torch.load(path, map_location=new["device"], weights_only=True))
        actual_labels, probabilities = five._predict(replay_model, loader, new["device"])
        assert np.array_equal(actual_labels, validation.Label.to_numpy())
        metrics = {
            "model": name, "patients": len(validation),
            "auroc": five.roc_auc_score(actual_labels, probabilities),
            "auprc": five.average_precision_score(actual_labels, probabilities),
            "best_epoch": settings["best_epoch"],
            "epochs_run": len(old_history if name == "original" else new_history),
        }
        frame = validation[["Patient_ID", "Label", "Fold", "LVEF"]].copy()
        frame["model"] = name
        frame["Probability"] = probabilities
        replay_predictions.append(frame)
        if name == "new_fold0":
            paired = frame.merge(saved_new_predictions[["Patient_ID", "Probability"]], on="Patient_ID", suffixes=("_replay", "_saved"), validate="one_to_one")
            metrics["max_abs_difference_from_saved_predictions"] = (paired.Probability_replay-paired.Probability_saved).abs().max()
        replay_rows.append(metrics)
    pd.DataFrame(replay_rows).to_csv(OUT / "validation_replay_summary.csv", index=False)
    pd.concat(replay_predictions, ignore_index=True).to_csv(OUT / "validation_replay_predictions.csv", index=False)

    manifest_paths = [notebook_path, ROOT / "src/five_fold_validation.py", ROOT / "src/train.py",
                      ROOT / "src/ecg_model.py", ROOT / "src/data_loader.py",
                      ROOT / "data/processed/patient_4ch/development/processed_metadata.csv"]
    result = {
        "startup_checks": startup,
        "loss_check": loss_check,
        "checkpoint_hashes_before": hashes_before,
        "checkpoint_hashes_after": {str(path.relative_to(ROOT)): digest_file(path) for path in (OLD, NEW)},
        "source_and_metadata_hashes": {str(path.relative_to(ROOT)): digest_file(path) for path in manifest_paths},
        "environment": {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        },
        "historical_environment_recorded": False,
        "historical_input_tensor_hashes_available": False,
        "community_test_loaded": False,
        "models_trained": False,
    }
    assert result["checkpoint_hashes_before"] == result["checkpoint_hashes_after"]
    (OUT / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(startup, indent=2))
    print(json.dumps(loss_check, indent=2))
    print(pd.DataFrame(replay_rows).to_string(index=False))
    print("Audit saved; checkpoints unchanged; no training or test evaluation.")


if __name__ == "__main__":
    main()
