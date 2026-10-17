# Research system card (no trained model)

## Status
A tested data/export/evaluation scaffold, actually fitted synthetic-only TF-IDF/logistic CPU checkpoint, and synthetic NumPy pointer-head smoke. No Qwen checkpoint was downloaded, trained, evaluated, or deployed. No clinical performance is established.

## Intended use
Developing a human-governed custom text-evidence classifier for SI3/SI4/SI5 flags after codebook approval and ethically governed labeling. Outputs contain flags and required human-review status, not generated rationales.

## Not intended
Predicting future self-harm or suicide, diagnosing, assigning treatment/discharge/access decisions, autonomous crisis triage, establishing that someone is safe, or substituting for clinician assessment. No validated clinical scale is implemented.

## Training data and labels
Abstract fixtures and an 800-record synthetic generator (64 underlying statement families) ship here; the fitted CPU demo is not a Qwen checkpoint. Human data provenance, consent/permissions, annotation reliability and adjudication are unresolved. The current taxonomy is draft and blocks export/training. Unknown is not negative. The four-class NO_RISK/SI3/SI4/SI5 contract is confirmed; detailed codebook boundaries remain unresolved. Teacher annotations from the user are silver labels, not human gold.

## Architecture
Planned upstream boundary: pinned Kev with Qwen3.8-27B, candidate/choice pointer decision head. Default generated plan is new-head LoRA from raw Qwen. Released full-weight Kev initialization requires a different compatible full-fine-tuning path. The tiny NumPy smoke is not this architecture or a language-model approximation benchmark.

## Evaluation and limitations
Four independent subject/family partitions; calibration/test overlap checks; precision, recall, average precision, Brier and confusion counts. No real metrics available. Thresholds are empirically selected, not probability-calibrated. Missing confidence intervals, near-duplicate detection, prospective/temporal validation, out-of-distribution checks, demographic/language subgroup testing and deployment safety evaluation preclude clinical deployment. Every result requires human review, including abstentions. A threshold does not guarantee safety.

## Privacy and operations
No external APIs or telemetry. Real narratives and identifiers must remain outside Git. Pseudonymous IDs/hashes can still be sensitive. No automated alerts, treatment actions, or deployment are included. Require privacy/security/legal review appropriate to the actual setting before collecting or processing real patient data.
