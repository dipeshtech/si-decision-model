# Custom SI flags classifier research starter

A local, configurable workflow for a Kev-inspired Qwen decision classifier producing flags, not generated rationales. A synthetic-only CPU baseline is genuinely fitted locally. The optional Qwen target is **not trained**. This project is not a validated clinical scale or a verified reproduction of Jev TypeSafe. It does not predict future suicide, diagnose people, recommend treatment/discharge, or replace human assessment.

## What works now

- Strict canonical JSONL validation with explicit true / false / unknown evidence states.
- Exclusive or ordinal label configuration; optional independent binary-label export.
- Deterministic train / validation / calibration / test splits, disjoint by subject and connected conversation/template families, plus exact-text leakage checks.
- Source-verified Kev choice / noul export (blocked until the codebook is approved).
- Per-label empirical threshold selection, held-out evaluation, and provenance overlap rejection.
- Flags-only decision function with abstention and human review required on every output.
- An actually fitted synthetic TF-IDF/logistic CPU demonstration and a tiny, real NumPy pointer-head gradient smoke test on synthetic vectors. It does not load Qwen or train a language model.
- Unit tests and an offline-test GitHub Actions workflow.

## Quick start

Python 3.11+ and NumPy are sufficient. No GPU, credentials, model downloads, or hosted APIs are needed.

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
risk-classifier smoke
risk-classifier validate --taxonomy configs/si_draft.json --data examples/synthetic.jsonl
risk-classifier split --taxonomy configs/si_draft.json --data examples/synthetic.jsonl --output artifacts/splits
```

Without installing this package, prefix commands with `PYTHONPATH=src python -m risk_classifier.cli` instead of `risk-classifier`.

## Taxonomy requires decisions before training

`configs/si_draft.json` records the confirmed four-class order NO_RISK=0, SI3=1, SI4=2, SI5=3 with highest-supported-severity intent. These are custom labels, not standardized clinical category definitions. The exclusive output contract is confirmed; approve the actual tuned codebook, timing/context boundaries, and ambiguous evidence rules before real training. Unmentioned plan or means is **unknown**, not an explicit denial. `unknown` annotations are excluded from supervised export; inference can output `INSUFFICIENT_EVIDENCE`.

Export and training-plan generation require `status: "approved"`, no unresolved questions, a definition for every label, and for exclusive/ordinal mode an explicit `no_risk_label` included in labels. A name such as `NONE` must mean no qualifying evidence **in the supplied text**, never that a person is safe. Approval is a human codebook decision; do not flip it merely to bypass validation. Ordinal mode exports named choices, not numeric intervals, and currently uses the same classification loss/metrics as exclusive mode.

The small original fixture file contains abstract arbitrary labels; the optional 800-record generator uses explicit, non-operational illustrative text and draft rules. Both are synthetic plumbing fixtures, not human gold or clinically validated supervision. Never use them to train a usable model or report model quality.

## Data format

```json
{"id":"opaque-example-id","subject_id":"opaque-person-id","conversation_id":"opaque-session-id","text":"Locally governed input text","label":"unknown","evidence":{"plan":"unknown","means":"unknown","timing":"unknown"}}
```

Each exclusive/ordinal record must have exactly one approved label or `unknown`. In multilabel mode replace `label` with `labels`, mapping every configured label to the string `true`, `false`, or `unknown`. Optional `template_family` groups synthetic/paraphrased examples. Identifiers should be pseudonymous. Split original records **before** paraphrasing/augmentation; reuse family IDs. Near-duplicate detection and time-based splits are not implemented: audit these separately, and reserve a future-time holdout with context cut off at prediction time. At least eight independent subject/family components are needed; small datasets may lack classes in held-out partitions. The tool does not invent class coverage.

## Training boundary

After the approved codebook and governed human labels exist:

```sh
risk-classifier export-kev --taxonomy configs/approved.json --data artifacts/splits/train.jsonl --output artifacts/kev-train.jsonl
risk-classifier train-plan --taxonomy configs/approved.json --data artifacts/splits/train.jsonl
```

The plan prints a reviewed argv template only. Training is an explicit manual step in a separately reviewed, pinned upstream checkout. It does not provision GPUs, download weights, install upstream code, or execute a job. The default plan is **raw Qwen LoRA with a new pointer head**, not a released Kev warm start. Released Kev-27B v2 contains full weights and cannot initialize ordinary LoRA with `--init_from`. Matching full fine-tuning is an alternative requiring a pinned checkpoint/config and appropriate multi-GPU resources; the older v1 LoRA route is not configured because its exact revision has not been verified. See [upstream boundary](docs/upstream.md).

27B training has not run here. GPU fit, upstream imports, tokenizer behavior, and real model outputs are unverified. State truncation can remove decisive context; explicitly measure it and reject unsafe truncation rather than silently treating missing context as negative.

## Calibration and evaluation

Provide one prediction line per record, with probabilities ordered exactly as taxonomy labels:

```json
{"id":"opaque-example-id","probabilities":[0.1,0.2,0.3,0.4]}
```

Use four probabilities for the confirmed class order; the schema requires the vector length to match the taxonomy. Exclusive probabilities must sum to one. Use validation data for model/hyperparameter selection. Fit thresholds only on the separate calibration set, then evaluate once on an untouched test set:

```sh
risk-classifier calibrate --taxonomy configs/approved.json --data artifacts/splits/calibration.jsonl --predictions artifacts/calibration-predictions.jsonl --output artifacts/calibration.json
risk-classifier evaluate --taxonomy configs/approved.json --data artifacts/splits/test.jsonl --predictions artifacts/test-predictions.jsonl --calibration artifacts/calibration.json --output artifacts/evaluation.json
```

This routine selects empirical per-label precision thresholds; it does **not** calibrate probabilities. Unsupported labels receive no threshold and abstain. The default 0.9 empirical precision target is an engineering placeholder, not a clinical standard or future guarantee. Temperature scaling, grouped bootstrap confidence intervals, subgroup audits and drift evaluation remain required research work. Report prevalence, per-class support/precision/recall, PR metrics, Brier score, confusion counts and abstention coverage. Calibration/test record, subject, exact-text and family overlap is rejected. Metadata hashes still need private storage. Train/validation separation must also be audited via the split manifest/data; prediction files cannot independently prove model provenance.

`flag` returns only `{"flag": "SI3" | "SI4" | "SI5" | approved_other_label | "INSUFFICIENT_EVIDENCE", "human_review_required": true}`. No automatic reassurance, care routing, alerts to third parties, or deployment is implemented. See [evaluation design](docs/evaluation-design.md) and [model card](MODEL_CARD.md).

## Privacy

Keep real data in access-controlled storage outside this repository. `.gitignore` excludes data, artifacts, checkpoints, logs, and common secrets; it is not a data-loss prevention system. Never commit patient narratives, credentials, identifiers, or generated examples derived from private records. Inspect staged changes before publication. No telemetry or external inference calls are implemented. Governance, lawful permissions, retention policy, clinician annotation/adjudication and deployment approval remain prerequisites.


## Actually trained low-compute demonstration

The CPU baseline is the primary runnable no-GPU path. Qwen3.8-27B remains an optional future GPU path; this baseline does not claim to match Qwen or Kev.

```sh
python -m pip install -e '.[baseline]'
python -m risk_classifier.baseline --taxonomy configs/si_draft.json --output artifacts/my-synthetic-demo
```

This generates 800 records from 64 non-operational statement templates (16 per class), groups repetitions of each underlying template together, trains TF-IDF plus logistic regression on 501 records, and saves JSON vocabulary/IDF/coefficients with taxonomy/hash. Validation/calibration/test contain 98/99/102 records in the verified seed-42 run, with all four classes represented. It reconstructs the JSON checkpoint, predicts, selects thresholds, and records synthetic-only integration metrics. No pickle is loaded. Class-stratified splitting requires at least eight homogeneous independent groups per class. The synthetic runner has 64 groups; unsupported labels still abstain. The scenario IDs are superficial synthetic markers. These fixtures are intentionally simple and do not establish broad instruction coverage or clinical generalization. Synthetic metric output is a plumbing check, not a quality benchmark. The demonstration evaluation explicitly enables flags only inside its synthetic-only runner; normal CLI evaluation remains gated by codebook approval.

## Import the teacher-labeled dataset

Teacher-generated annotations are silver labels, not human gold. No private source transcripts or actual deployment codebook are in this repository. Keep the full codebook once in governed local storage and refer to its version and SHA-256; the fixed-feature baseline does not repeat the prompt as feature text.

```sh
python -m risk_classifier.teacher_import --input private_data/teacher.jsonl --taxonomy configs/si_draft.json --codebook private_data/codebook.txt --codebook-version actual-v1 --output artifacts/teacher-canonical.jsonl
```

Input fields: example_id, participant_group, conversation_group, text, teacher_label; optional human_label, teacher_model, teacher_prompt_version, evidence, duplicate_cluster, template_family, source_type, context_cutoff and timestamp. Human/teacher/target labels remain separate; a supplied human label is selected as target and disagreements require review. Synthetic input must name its family. Genuine four-class teacher_probabilities require probability_provenance; verbal confidence is not treated as probabilities. The importer hashes the codebook but does not interpret or claim coverage of its contents. The importer does not itself grant permission to process or upload private data.

For Qwen, compare a fixed short instruction with a full-codebook instruction under controlled, identical splits. Full prompts are possible, but cost more tokens. Do not assume a concise instruction preserves the long codebook until evaluated. Kev export currently embeds taxonomy question/criteria per question as its source format requires; it does not magically eliminate model input tokens through a codebook reference. A manifest hash alone does not teach a model the codebook.


## Rule coverage audit

```sh
python -m risk_classifier.coverage --manifest configs/sample_rules.json --examples examples/sample_challenges.jsonl
```

The shipped manifest is SAMPLE ONLY, not the unavailable full 2–3 page codebook. It intentionally reports gaps. Map each actual instruction to a stable rule_id and version; supply multiple reviewed positive, negative, boundary and context cases, interaction cases, and independently held-out challenge families. The command reports missing case counts, missing/overlapping challenge families, unreviewed/unresolved counts and unknown rule IDs. Counts never prove semantic coverage or exhaustive compliance. Challenge examples contrast syllabus/KMS hyperbole with endorsed current ideation and ambiguous context; the phrase itself never forces NO_RISK. These sample annotations still need human adjudication.

## Configurable training plan and local inference

`configs/training.json` exposes the verified upstream learning rates, LoRA rank, pointer dimension, batch/accumulation, epochs, context length, seed and checkpointing settings. Pass `--training-config configs/training.json` to `train-plan`. Unsupported/unverified settings fail rather than silently changing the model family. Qwen3.8-27B is the pinned optional GPU target; no official smaller Qwen3.8 checkpoint is assumed. QLoRA and alternative LoRA module selection require a separate compatibility check for this hybrid backbone.

```sh
python -m risk_classifier.predict --model artifacts/my-synthetic-demo/baseline.json --thresholds artifacts/my-synthetic-demo/thresholds.json --input examples/synthetic.jsonl --output artifacts/demo-flags.jsonl
```

This emits only an ID, flag and human-review boolean. Synthetic-trained or unapproved models always abstain in this inference entrypoint, even if the synthetic integration evaluation has measurable class predictions. Real approved data, reviewed thresholds and further deployment validation are required before changing that policy.

## Configurable LLM synthetic generation (no paid calls performed)

```sh
# Default: plan only, no file writes and no network.
python -m risk_classifier.llm_generate --config configs/generation.json --output artifacts/mock-generated.jsonl
# Explicit offline mock generation:
python -m risk_classifier.llm_generate --config configs/generation.json --output artifacts/mock-generated.jsonl --execute
```

The adapter supports OpenAI-compatible chat-completion endpoints with configurable provider/model/base URL, named credential environment variable, codebook/rule instructions, per-class/per-rule quotas, case-type distributions, length bounds, difficulty, seed and family IDs. Mock output is abstract plumbing text, not high-quality supervision. Generated rows are unreviewed/unresolved. Exact normalized duplicates are rejected; near-duplicate/semantic review remains a separate requirement. Budget reservation and resume provenance are persisted locally, and uncertain failures are not automatically retried.

`configs/generation_live_template.json` is deliberately blocked: select a verified provider model, enter actual input/output pricing and an explicit approved dollar limit, then review endpoint/data-sharing scope before `--execute`. Set credentials only in the named environment variable, never a config or repository file. No real provider calls were run. Only public synthetic instructions are permitted by this adapter; do not put private patient transcripts in prompts. Configured cost estimates/reservations are conservative accounting controls, not a provider billing guarantee. Use provider-side spend limits as well. An endpoint must be HTTPS and its host explicitly listed; redirects are rejected. The adapter validates structured output and reported usage. Real-provider compatibility remains unverified.

Split and dedup controls live in the data workflow: group by participant/conversation/duplicate/template lineage before augmentation; optional homogeneous-label stratification fails if independent group support is insufficient. Evaluation thresholds and approved-taxonomy abstention remain separate from generation. Additional unimplemented model/provider knobs must be explicitly added and tested rather than silently accepted as working features.

## Long-transcript windowing without label leakage

Split whole participants/conversations/template families first. `risk_classifier.windows` then takes canonical JSONL with an explicit split and pinned tokenizer_revision plus actual tokenizer-produced character token_offsets (`[[start,end], ...]`). It does not download a tokenizer or pretend word counts are model tokens.

```sh
python -m risk_classifier.windows --input artifacts/split-tokenized.jsonl --output artifacts/windows.jsonl --max-input-tokens 2048 --overlap 256 --reserved-tokens 256
```

2048/256 are configurable example settings, not a universal model limit. Reserve the actual system/instruction/choice/special-token overhead and check the target tokenizer's final serialized input length; a 128/512-token encoder needs its own smaller settings. Content capacity equals max-input-tokens minus reserved-tokens. Offsets, original ID, subject/conversation/family, split, tokenizer revision and budget are preserved. Uncovered non-whitespace input fails instead of silent truncation. Role markers remain text, but a token window may cut a turn: these windows require context-aware human/teacher labeling before use. Conversation-aware boundary alignment is not implemented.

Every window target is unknown, regardless of the transcript label. The transcript label is retained only in a clearly named provenance field. Never copy one whole-transcript label to every window. No window aggregation, multiple-instance learning, majority vote, or highest-window severity inference is implemented; those need separate design/validation. Unknown windows are excluded from Kev supervised export. Group overlap is rejected before windowing, including repeated template variants. The exporter does not infer missing means, plan or timing from a cropped window.

The CPU baseline accepts `--settings configs/baseline.json` for vocabulary size, minimum document frequency, n-gram extent, logistic regularization, optimization iterations and split/model seed. Unknown options fail; feature parameters are persisted and reconstructed at inference. Keep settings fixed before touching the test set.

Automatic preprocessing is available with an explicitly selected local fast tokenizer (optional Transformers dependency). No manual offsets are needed:

```sh
python -m risk_classifier.tokenize_windows --splits artifacts/splits --tokenizer /path/to/local/tokenizer --tokenizer-revision VERIFIED_REVISION --output artifacts/token-windows --max-input-tokens 128 --overlap 16 --reserved-tokens 16
```

This loads local files only, disables remote code, tokenizes without truncation, checks the tokenizer limit/special-token reserve, creates offsets automatically, and re-tokenizes each window to verify its actual budget. All four split files are checked together for group leakage; raw files stay untouched. Tokenizer fingerprints are saved. Choose the actual model context budget; 128 here is an example for a small encoder, not a Qwen limit. Full serialized instruction/choice overhead remains the caller's responsibility. Aggregation/MIL is still unimplemented and every produced window remains unlabelled/unknown until annotated.

## Genuine small pretrained neural CPU run

`optional/cpu-neural/` includes a real, completed BERT-tiny full-encoder fine-tuning experiment: 4.39 million trainable parameters, CPU only, same frozen synthetic splits, verified safetensors checkpoint reload and changed encoder weights. It is not Qwen. Training plus epoch validation took 19.41 seconds in the measured environment, excluding setup/download/loading. Synthetic test SI5 recall was only 48% (13 of25 SI5 cases predicted SI3), so it is plainly unsuitable for clinical deployment. Operational inference always abstains and requires human review. The script, pinned model/dependency provenance and aggregate results are in source; weights are a separate artifact. See its README for reproduction and limitations. No live patient or teacher data was used.

Generation ledgers are single-process only: run one process per output/ledger. No interprocess lock is implemented, so concurrent invocations must not share a budget ledger.
