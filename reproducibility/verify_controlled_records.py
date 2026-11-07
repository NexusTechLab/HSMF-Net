"""Verify the 35 controlled prediction records; no image data, weights or training."""
from pathlib import Path
import csv, hashlib, json, math

ROOT = Path(__file__).resolve().parents[1]

def main():
    records = json.loads((ROOT / "reproducibility/controlled_run_records.json").read_text(encoding="utf-8"))
    counts = {}
    for item in records:
        path = ROOT / item["prediction_file"]
        metric = json.loads((ROOT / item["metrics_file"]).read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != metric["predictions_sha256"]:
            raise ValueError("Prediction hash mismatch: " + item["run"])
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 570 or len({r["sample_id"] for r in rows}) != 570:
            raise ValueError("Unexpected rows or duplicate sample IDs: " + item["run"])
        cm = [[0] * 4 for _ in range(4)]
        for row in rows:
            y, prediction = int(row["true_label"]), int(row["pred_label"])
            if not (0 <= y < 4 and 0 <= prediction < 4):
                raise ValueError("Invalid label: " + item["run"])
            cm[y][prediction] += 1
        if [sum(row) for row in cm] != [135, 150, 150, 135]:
            raise ValueError("Class-support mismatch: " + item["run"])
        correct = sum(cm[i][i] for i in range(4))
        f1 = []
        for i in range(4):
            tp = cm[i][i]
            fp = sum(row[i] for row in cm) - tp
            fn = sum(cm[i]) - tp
            f1.append(200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        if cm != metric["confusion_counts"] or correct != metric["correct"]:
            raise ValueError("Confusion/count mismatch: " + item["run"])
        if not math.isclose(100 * correct / 570, metric["accuracy_pct"], abs_tol=1e-10):
            raise ValueError("Accuracy mismatch: " + item["run"])
        if not math.isclose(sum(f1) / 4, metric["macro_f1_pct"], abs_tol=1e-10):
            raise ValueError("Macro-F1 mismatch: " + item["run"])
        counts[item["study"]] = counts.get(item["study"], 0) + 1
    if counts != {"matched24": 25, "spa_uniform24": 10}:
        raise ValueError("Unexpected study counts: " + repr(counts))
    print(json.dumps({"passed": True, "studies": counts, "predictions_per_run": 570,
                      "image_data_loaded": False, "historical_input_identities_authenticated": False}, indent=2))

if __name__ == "__main__":
    main()
