"""Development-only threshold selection; held-out evaluation; flags, no rationale."""
import math
import numpy as np


def probabilities(values, size):
    p = np.asarray(values, dtype=float)
    if p.shape != (size,) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid probability vector")
    return p


def validate_predictions(rows, labels, exclusive=True):
    seen = set()
    for row in rows:
        if not isinstance(row.get("id"), str) or row["id"] in seen:
            raise ValueError("Prediction IDs must be unique strings")
        seen.add(row["id"])
        p = probabilities(row.get("probabilities"), len(labels))
        if exclusive and not np.isclose(p.sum(), 1.0, atol=1e-6):
            raise ValueError("Exclusive probabilities must sum to one")
    return rows


def select_thresholds(records, predictions, labels, minimum_precision=0.9):
    """Fit thresholds ONLY on a separate calibration split; unsupported labels abstain.

    This empirical target is not a guarantee of future precision or safety.
    Scores are not recalibrated by this routine.
    """
    if not 0 < minimum_precision <= 1:
        raise ValueError("Minimum precision must be in (0,1]")
    truth = {r["id"]: r for r in records}
    if set(truth) != {p["id"] for p in predictions}:
        raise ValueError("Prediction IDs must exactly match calibration record IDs")
    validate_predictions(predictions, labels)
    thresholds = {}
    for j, label in enumerate(labels):
        known = [(p["probabilities"][j], truth[p["id"]]["label"] == label)
                 for p in predictions if truth[p["id"]]["label"] != "unknown"
                 and int(np.argmax(p["probabilities"])) == j]
        if not known or not any(y for _, y in known):
            thresholds[label] = None
            continue
        accepted = []
        for cutoff in sorted({score for score, _ in known}):
            chosen = [y for score, y in known if score >= cutoff]
            precision = sum(chosen) / len(chosen)
            if precision >= minimum_precision:
                accepted.append(cutoff)
        thresholds[label] = min(accepted) if accepted else None
    return thresholds


def flag(scores, labels, thresholds, approved=False):
    p = probabilities(scores, len(labels))
    if not np.isclose(p.sum(), 1, atol=1e-6):
        raise ValueError("Exclusive probabilities must sum to one")
    index = int(np.argmax(p))
    label = labels[index]
    cutoff = thresholds.get(label)
    if cutoff is not None and (not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff) or not 0 <= cutoff <= 1):
        raise ValueError("Invalid threshold")
    selected = label if approved and cutoff is not None and p[index] >= cutoff else "INSUFFICIENT_EVIDENCE"
    # Every output requires human review until separate deployment validation.
    return {"flag": selected, "human_review_required": True}


def evaluate(records, predictions, labels, thresholds, approved=False):
    truth = {r["id"]: r for r in records}
    validate_predictions(predictions, labels)
    if set(truth) != {p["id"] for p in predictions}:
        raise ValueError("Prediction IDs must exactly match evaluation record IDs")
    confusion = {a: {b: 0 for b in labels + ["INSUFFICIENT_EVIDENCE"]} for a in labels}
    brier, unknown, emitted, total = 0.0, 0, 0, 0
    scores_by_label = {label: [] for label in labels}
    for row in predictions:
        actual = truth[row["id"]]["label"]
        result = flag(row["probabilities"], labels, thresholds, approved)
        emitted += result["flag"] != "INSUFFICIENT_EVIDENCE"
        if actual == "unknown":
            unknown += 1
            continue
        total += 1
        confusion[actual][result["flag"]] += 1
        for j, label in enumerate(labels):
            target = int(actual == label)
            brier += (row["probabilities"][j] - target) ** 2
            scores_by_label[label].append((row["probabilities"][j], target))
    metrics = {}
    for label in labels:
        tp = confusion[label][label]
        predicted = sum(confusion[a][label] for a in labels)
        support = sum(confusion[label].values())
        # Average precision, grouped by tied score, avoids arbitrary tie ordering.
        ranked = scores_by_label[label]
        positive = sum(y for _, y in ranked)
        ap, previous_recall = 0.0, 0.0
        for cutoff in sorted({s for s, _ in ranked}, reverse=True):
            selected = [y for s, y in ranked if s >= cutoff]
            recall = sum(selected) / positive if positive else 0
            ap += (recall - previous_recall) * sum(selected) / len(selected)
            previous_recall = recall
        metrics[label] = {"precision": tp / predicted if predicted else None,
                          "recall": tp / support if support else None,
                          "support": support, "average_precision": ap if positive else None}
    return {"known_records": total, "unknown_records": unknown,
            "coverage": emitted / len(predictions) if predictions else None,
            "multiclass_brier": brier / total if total else None,
            "confusion": confusion, "per_class": metrics,
            "warning": "Research only; no clinical performance or safety claim. Thresholds must be fitted on a disjoint calibration split."}
