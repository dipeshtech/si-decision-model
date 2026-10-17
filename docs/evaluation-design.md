# Research-only SI text classifier: evaluation checklist

## Scope and labels (project design recommendations)

This is a custom annotation taxonomy for evidence in supplied text, not a standardized clinical scale, diagnosis, or prediction of suicide. Preserve the user's definitions: SI3 ideation/wish to end life without plan/means; SI4 intent/known method or plan; SI5 means plus plan plus specific timing. Publish a versioned annotation manual defining time window, speaker, negation, quotation, hypothetical statements, and precedence among labels. Resolve whether “without” requires explicit denial; do not silently equate unmentioned details with absence.

Include a no-qualifying-SI label (NO_RISK) and an insufficient-evidence/abstention outcome. NO_RISK means the input does not establish qualifying content, never “this person is safe.” Insufficient evidence means the supplied text cannot support a definite taxonomy label. Otherwise a three-class classifier is forced to flag ordinary text and cannot express ambiguity. Missing plan/means/timing are unknown, not false; retain present/absent/unknown annotation states if evidence attributes are recorded. Distinguish annotation insufficiency from model low confidence in metadata, even if both route to review.

Use independently reviewed gold labels and adjudication, recording disagreement rather than replacing it with unreviewed teacher outputs. Include no-SI, unclear statements, historical versus current content, negation, third-person reports, quotations, and conflicting context. Synthetic examples test software plumbing only; they cannot establish clinical validity or real-world performance. No real patient data ingestion or clinical deployment is authorized by this scaffold.

## Leakage and test design

- Assign complete participants to one split; keep conversations and all overlapping windows together. Group duplicate/paraphrased examples and synthetic template families too. Where identifiers connect examples, group by the connected component rather than a convenient row ID.
- Predefine a prediction timestamp and exclude later turns, assessments, outcomes, or post-event metadata. Add a later-time holdout and, when appropriate, a distinct source/site holdout. Describe which population and time period each test represents.
- Fit transformations, sampling, hyperparameters, calibration, and thresholds on training/development data only. Lock the final test; do not repeatedly tune against it.

TRIPOD+AI explicitly calls for evaluation data independent of development, no participant overlap, and representative evaluation populations. These text-specific grouping and timestamp rules apply that principle to this project. [TRIPOD+AI, BMJ 2024](https://www.bmj.com/content/385/bmj-2023-078378)

## Metrics, uncertainty, and thresholds

Report label supports, prevalence, complete confusion counts, and per-class precision/recall/F1 plus macro summaries. Add one-versus-rest PR curves and average precision, with the exact computation named; retain AUROC as complementary information. Evaluate on plausible target prevalence, not only an artificially balanced test. PR/precision depends on class prevalence, making it especially informative for rare flags. [Saito & Rehmsmeier, PLOS ONE 2015](https://doi.org/10.1371/journal.pone.0118432)

Report participant-cluster bootstrap confidence intervals and rare-class counts; do not claim precision from tiny supports. A class absent from a test set has undefined class-specific discrimination, not zero performance. Include review/abstention rate and coverage, performance on retained cases, and errors among abstained cases. Test important language/source/context slices where support permits, with uncertainty.

If scores exist, report reliability plots, Brier score and log loss; evaluate calibration separately from accuracy. Fit any temperature scaling on held-out development predictions. Softmax scores and generated verbal confidence are not automatically calibrated probabilities. [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)

Select operating thresholds from a predeclared research objective on development data, accounting for review workload and false negatives/positives; report the entire trade-off rather than asserting a universal safe cutoff. Keep probabilities as internal evaluation metadata if the user-facing contract is flags-only. No generated reasoning is necessary: fixed labels plus auditable annotation provenance suffice.

## Use boundary

Require human review of research outputs. Do not trigger treatment, discharge, emergency outreach, restrictions, or other consequential actions automatically. NICE specifically advises against using risk scales to predict suicide/self-harm or determine treatment/discharge; this supports a strict separation between an experimental text label and clinical decisions. [NICE NG225, recommendations 1.6.1–1.6.6](https://www.nice.org.uk/guidance/ng225/chapter/Recommendations#risk-assessment-tools-and-scales)
