import argparse
import hashlib
import json
from pathlib import Path
from .data import taxonomy, load_records, split_records, write_jsonl, read_json, assert_disjoint
from .evaluation import select_thresholds, evaluate
from .smoke import run


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Research-only flags classifier tooling")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")
    for name in ("validate", "split", "export-kev", "calibrate", "evaluate", "train-plan"):
        p = sub.add_parser(name)
        p.add_argument("--taxonomy", required=True)
        p.add_argument("--data", required=True)
        if name not in {"validate", "train-plan"}:
            p.add_argument("--output", required=True)
        if name in {"calibrate", "evaluate"}:
            p.add_argument("--predictions", required=True)
        if name == "calibrate":
            p.add_argument("--minimum-precision", type=float, default=0.9)
        if name == "evaluate":
            p.add_argument("--calibration", required=True)
        if name == "train-plan":
            p.add_argument("--training-config")
        if name == "split":
            p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if args.command == "smoke":
            print(json.dumps(run(), indent=2)); return
        spec = taxonomy(args.taxonomy)
        rows = load_records(args.data, spec)
        if args.command == "validate":
            result = {"records": len(rows), "subjects": len({r["subject_id"] for r in rows}), "taxonomy_status": spec.get("status")}
        elif args.command == "split":
            result = split_records(rows, args.seed)
            output = Path(args.output)
            if output.exists() and any(output.iterdir()):
                raise ValueError("Refusing to overwrite a nonempty split directory")
            for name, records in result.items():
                write_jsonl(output / f"{name}.jsonl", records)
            result = {name: len(records) for name, records in result.items()}
        elif args.command == "export-kev":
            from .kev import export_records
            exported, skipped = export_records(rows, spec)
            write_jsonl(args.output, exported)
            result = {"exported": len(exported), "skipped_unknown": skipped}
        elif args.command == "train-plan":
            from .kev import train_plan
            result = train_plan(args.data, spec, read_json(args.training_config) if args.training_config else None)
        else:
            if spec["mode"] == "multilabel":
                raise ValueError("Calibration/evaluation CLI currently supports exclusive or ordinal mode only")
            predictions = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
            if args.command == "calibrate":
                result = {"kind": "empirical_threshold_selection", "taxonomy_hash": digest(spec),
                          "thresholds": select_thresholds(rows, predictions, spec["labels"], args.minimum_precision),
                          "minimum_precision": args.minimum_precision,
                          # No narratives in calibration artifact; use hashes for overlap detection.
                          "record_hashes": [digest(r["id"]) for r in rows],
                          "subject_hashes": sorted({digest(r["subject_id"]) for r in rows}),
                          "text_hashes": [digest(r["text"].strip()) for r in rows],
                          "family_hashes": [digest([key, r[key]]) for r in rows for key in ("conversation_id", "template_family", "duplicate_cluster") if r.get(key)]}
            else:
                calibration = read_json(args.calibration)
                if calibration.get("taxonomy_hash") != digest(spec):
                    raise ValueError("Calibration taxonomy mismatch")
                for artifact_key, current in (
                    ("record_hashes", [digest(r["id"]) for r in rows]),
                    ("subject_hashes", [digest(r["subject_id"]) for r in rows]),
                    ("text_hashes", [digest(r["text"].strip()) for r in rows]),
                    ("family_hashes", [digest([key, r[key]]) for r in rows for key in ("conversation_id", "template_family", "duplicate_cluster") if r.get(key)])):
                    if artifact_key not in calibration:
                        raise ValueError("Calibration provenance missing")
                    if set(current) & set(calibration[artifact_key]):
                        raise ValueError("Calibration/evaluation leakage detected")
                from .kev import require_approved
                try:
                    require_approved(spec)
                    approved = True
                except ValueError:
                    approved = False
                result = evaluate(rows, predictions, spec["labels"], calibration["thresholds"], approved)
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n")
            result = {"written": str(path)}
        print(json.dumps(result, indent=2))
    except (ValueError, KeyError, OSError, TypeError) as error:
        # Do not print raw records or source text.
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
