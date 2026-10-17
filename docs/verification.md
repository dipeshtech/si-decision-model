# Local verification, 2026-10-17

- Python unittest discovery: 42 tests passed, including data/schema, subject/family splits, threshold argmax regression, calibration leakage, checkpoint reconstruction, draft abstention, rule coverage, eight LLM-generation safety/resume tests, and five token-window/budget/leakage tests and two automatic tokenizer-adapter tests.
- Real CPU TF-IDF/logistic model fitted on 501 synthetic training records and serialized/reloaded as JSON. No pickle execution.
- Final generated dataset: 800 examples, 64 core template families (16 per class); train/validation/calibration/test 501/98/99/102. All four classes have support in every partition. Families are disjoint; synthetic wording remains simple and does not prove real-world generalization.
- Synthetic integration evaluation: four numeric thresholds, 57/102 emitted flags (55.9% coverage). NO_RISK test cases all abstain at the selected threshold. These are plumbing observations, not clinical model quality claims.
- Public inference entrypoint: 24 abstract fixtures produced abstentions with human review required, because the model is synthetic-only and taxonomy remains unapproved.
- Tiny NumPy pointer-head backward/update smoke: loss 1.0983726315 -> 1.0895093292; gradient norm 0.4229584949. This is not a Qwen/Kev forward/backward pass.
- LLM generation dry-run: seven planned examples, zero network calls and no output write. Explicit mock run: seven examples, zero network calls and $0 configured spend. Live-provider transport exercised only by injected fake responses.
- Python compile checks passed.

## Not run

No real-provider LLM calls, GPU jobs, Qwen/Kev downloads, Qwen/Kev training, upstream runtime integration, real teacher-data training, clinical validation or deployment. GitHub CI status must be checked separately after publication. No original public corpus or private transcript is included in the source package.

## Reproduce

```sh
python -m pip install -e '.[baseline]'
python -m unittest discover -s tests -v
python -m risk_classifier.cli smoke
python -m risk_classifier.baseline --taxonomy configs/si_draft.json --output artifacts/new-demo
python -m risk_classifier.llm_generate --config configs/generation.json --output artifacts/new-mock.jsonl
python -m risk_classifier.coverage --manifest configs/sample_rules.json --examples examples/sample_challenges.jsonl
```

Choose fresh output paths. Thresholds and synthetic metrics are deterministic for the checked environment/seed; numerical differences can occur across library versions. No semantic coverage guarantee follows from rule/example counts.

## Additional actual CPU neural experiment

A separate 4,386,436-parameter BERT-tiny encoder was genuinely fine-tuned on CPU with the same frozen synthetic splits. See optional/cpu-neural/README.md and results/ for reproducible source and aggregate evidence. Two checkpoint reloads matched; an encoder weight changed by max0.017396. The synthetic SI5 recall was only48%, so this model is unsuitable for safety deployment. Operational inference always abstains. It is not Qwen. No test-set tuning/retraining was performed after the frozen run.

Automatic preprocessing was then run with the locally saved real BERT fast tokenizer, without loading weights, downloading, or remote code. All800 records produced unknown-labelled windows at128 total tokens,16 overlap,16 reserved, preserving grouped splits. These short fixtures do not validate long-transcript understanding.
