# Independent offline review

Reviewed and reran on 2026-10-17. Research scaffold only; no clinical validation.

## Verified

- 40 unittest cases pass in the available local environment.
- An independent 800-record synthetic baseline run completed with 501/98/99/102 train/validation/calibration/test records in the final saved run.
- Actual underlying statement templates, not merely family IDs, were checked: all 64 statement templates are disjoint across partitions.
- Missing labels remain invalid; explicit unknown labels are distinct from NO_RISK and are excluded from supervised export.
- JSON-persisted TF-IDF/logistic predictions matched a separately refit sklearn reference exactly (maximum absolute probability difference 0.0).
- Unsupported CLI training arguments are rejected rather than silently ignored.
- Calibration now selects thresholds only among argmax-eligible predictions, matching flag emission. A reproduced counterexample is covered by regression testing.
- Calibration/test overlap checks and draft abstention tests pass. Non-abstaining CLI evaluation requires the complete codebook-approval gate.
- Checkpoint storage is JSON, not pickle. Saved taxonomy integrity and fixed feature parameters are checked; nonfinite coefficient/intercept values are rejected.
- Windowing checks require preassigned disjoint groups, actual tokenizer offsets and a tokenizer revision; reject uncovered/truncated text and invalid token/overlap budgets; and always output unknown window labels rather than inheriting transcript supervision. Five window tests pass. Real-tokenizer integration was not run.
- The workflow now installs the baseline optional dependency required by tests. Remote GitHub Actions was not run.

## Material limitations

The final synthetic run includes all four classes in every split and 16 underlying statement templates per class (64 total). All four thresholds are numeric. Independently recomputed test coverage is 57/102 (55.88%); all 25 NO_RISK test records abstain. This replaces the earlier sparse-coverage fixture run. No accuracy or safety claim follows from these simplistic fixtures. The synthetic evaluation intentionally exercises decisions with an explicit research override; it does not approve the draft codebook for real use.

No real teacher transcript dataset was available for this review. No Qwen/Kev model weights were downloaded or trained, no upstream training code was executed, and no GPU fit was tested. Real-data governance, codebook approval, independent human gold labels, full class/context coverage, uncertainty estimates, near-duplicate/temporal audits and prospective validation remain prerequisites.


## Additional adapter and neural checks

The LLM generation adapter was reviewed and its eight offline tests pass, including dry-run/no-network behavior, bounded accounting, uncertain-call reservation retention, resume idempotence, response validation and duplicate rejection. No live provider request was made in this review. Configured prices and conservative reservations are not a provider billing guarantee; use one process per output/ledger because interprocess locking is not implemented.

A separate compact BERT CPU experiment was source-reviewed. Epoch selection uses validation macro-F1 only; thresholds use calibration data only; the test evaluation occurs after checkpoint and thresholds are fixed. This is a different model from Qwen/Kev. The existing verification script was independently run in its installed environment: checkpoint reload and deterministic predictions across two reloads pass, a pretrained encoder tensor changed by maximum absolute difference 0.0173959993, operational outputs abstain, and empty/overlength inputs are rejected appropriately. No retraining or held-out test rerun was performed during this independent verification. Reported synthetic test metrics remain nonclinical; source inspection and checkpoint verification do not establish real-world model validity.
