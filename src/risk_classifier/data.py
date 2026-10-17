"""Strict annotation validation and deterministic, subject-disjoint splits."""
import hashlib
import json
import random
from pathlib import Path

STATES = {"true", "false", "unknown"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def taxonomy(path):
    spec = read_json(path)
    labels = spec.get("labels")
    if not isinstance(labels, list) or not labels or any(not isinstance(x, str) or not x.strip() for x in labels):
        raise ValueError("Taxonomy requires nonempty string labels")
    if len(set(labels)) != len(labels) or "unknown" in labels:
        raise ValueError("Duplicate or reserved taxonomy labels")
    if spec.get("mode") not in {"exclusive", "ordinal", "multilabel"}:
        raise ValueError("Unsupported taxonomy mode")
    if not isinstance(spec.get("question"), str) or not spec["question"].strip():
        raise ValueError("Taxonomy requires a question")
    return spec


def validate(records, spec):
    seen = set()
    for row in records:
        for key in ("id", "subject_id", "text"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Each record requires nonempty {key}")
        if row["id"] in seen:
            raise ValueError("Duplicate record ID")
        seen.add(row["id"])
        if spec["mode"] == "multilabel":
            labels = row.get("labels")
            if not isinstance(labels, dict) or set(labels) != set(spec["labels"]):
                raise ValueError("Every label must have an explicit state")
            if any(not isinstance(v, str) or v not in STATES for v in labels.values()):
                raise ValueError("States must be true, false or unknown strings")
        elif row.get("label") not in spec["labels"] + ["unknown"]:
            raise ValueError("Exclusive/ordinal records require an explicit label or unknown")
        evidence = row.get("evidence", {})
        if not isinstance(evidence, dict) or any(not isinstance(v, str) or v not in STATES for v in evidence.values()):
            raise ValueError("Evidence must use explicit true/false/unknown states")
    return records


def load_records(path, spec):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON at line {line_no}") from None
    if not records:
        raise ValueError("Dataset is empty")
    return validate(records, spec)


def split_records(records, seed=42, stratify=False):
    """Four disjoint subject sets: train, validation, calibration and test.

    Does not stratify or promise class coverage. Small datasets fail rather than
    silently reuse subjects. All sessions for one subject follow that subject.
    """
    # Connected components prevent shared conversation/template families leaking.
    parent = {r["subject_id"]: r["subject_id"] for r in records}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    links = {}
    for row in records:
        for key in ("conversation_id", "template_family", "duplicate_cluster"):
            if row.get(key):
                token = (key, row[key])
                if token in links:
                    parent[root(row["subject_id"])] = root(links[token])
                links[token] = row["subject_id"]
    groups = sorted({root(r["subject_id"]) for r in records})
    if len(groups) < 8:
        raise ValueError("At least eight distinct subjects required for four-way split")
    random.Random(seed).shuffle(groups)
    if stratify:
        by_class = {}
        for group in groups:
            values = {r["label"] for r in records if root(r["subject_id"]) == group}
            if len(values) != 1 or "unknown" in values:
                raise ValueError("Stratified mode requires homogeneous known-label groups")
            by_class.setdefault(next(iter(values)), []).append(group)
        assignments = {name:set() for name in ("train", "validation", "calibration", "test")}
        for label, members in by_class.items():
            if len(members) < 8:
                raise ValueError("Stratified mode requires eight independent groups per class")
            held = max(1,len(members)//6)
            chunks = [members[:-3*held],members[-3*held:-2*held],members[-2*held:-held],members[-held:]]
            for name, chunk in zip(assignments,chunks):
                assignments[name].update(chunk)
        result = {name:[r for r in records if root(r["subject_id"]) in selected] for name,selected in assignments.items()}
        assert_disjoint(result)
        return result
    n = len(groups)
    heldout = max(1, n // 8)
    boundaries = [0, n - 3 * heldout, n - 2 * heldout, n - heldout, n]
    result = {}
    for name, low, high in zip(("train", "validation", "calibration", "test"), boundaries, boundaries[1:]):
        selected = set(groups[low:high])
        result[name] = [r for r in records if root(r["subject_id"]) in selected]
    assert_disjoint(result)
    return result


def assert_disjoint(splits):
    owners = {}
    content_owners = {}
    linked_owners = {}
    for name, rows in splits.items():
        for row in rows:
            subject = row["subject_id"]
            if subject in owners and owners[subject] != name:
                raise ValueError("Subject leakage across splits")
            owners[subject] = name
            for key in ("conversation_id", "template_family", "duplicate_cluster"):
                if row.get(key):
                    token = (key, row[key])
                    if token in linked_owners and linked_owners[token] != name:
                        raise ValueError("Conversation or template-family leakage across splits")
                    linked_owners[token] = name
            digest = hashlib.sha256(row["text"].strip().encode()).hexdigest()
            if digest in content_owners and content_owners[digest] != name:
                raise ValueError("Identical text leakage across splits")
            content_owners[digest] = name


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
