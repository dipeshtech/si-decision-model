# CPU-only pretrained neural experiment

This is a real full-encoder fine-tune of Google's 4,386,436-parameter BERT-tiny, **not Qwen**, and not a frozen embedding/classifier-only run. It uses only the synthetic fixture corpus, with an unapproved custom four-class taxonomy. It is a reproducible pipeline demonstration, not a clinical classifier or deployment-ready model.

## Frozen first experiment
- Base model: `google/bert_uncased_L-2_H-128_A-2`; revision `30b0a37ccaaa32f332884b96992754e246e48c5f`; Apache-2.0.
- All 4,386,436 parameters trainable, 2 transformer layers, hidden size 128.
- CPU only; 4 threads; seed 20261017; AdamW, learning rate .0002, weight decay .01; batch 16; 12 epochs; max input 128 tokens.
- 501 train / 98 validation / 99 calibration / 102 test synthetic records. Template-family, synthetic-person and conversation IDs are pairwise disjoint across all splits. Exact normalized text duplicates are rejected. Pseudonymous fixture person IDs do not validate real person generalization.
- Select checkpoint on validation macro-F1; choose thresholds on calibration only; score test **once** after both fixed. No post-test tuning/retraining. Best epoch 8.
- Training + per-epoch validation: 19.41 seconds, excluding installation, model download, tokenization, initial loading and verification. Test inference 0.041 seconds for 102 short examples; warm in-process measurement, not end-to-end request latency.
- Held-out synthetic test: accuracy 70.59%, macro-F1 .7071. **SI5 recall 48% (12 of 25), with 13 SI5 examples incorrectly predicted SI3.** This fails any credible safety deployment goal.
- Threshold-based experimental coverage 73.53%; accepted accuracy 85.33%. Empirical calibration thresholds are not guarantees or posterior probability calibration. Operational prediction always abstains with human review.
- Architecture supports 512 positions, but experiment cap is 128. All synthetic records are at most 55 tokens. Long transcript understanding is untested; oversized inputs are rejected, not silently truncated. Window annotations must be locally valid; never propagate transcript SI5 label to every window.

## Reproduce
Use Python 3.12 and a new virtual environment. Install CPU PyTorch from the official CPU index, then the pinned requirements:

```sh
python -m venv .venv
.venv/bin/pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
bash download_pretrained.sh
.venv/bin/python train_cpu.py --data-dir ../../artifacts/synthetic-baseline-final --output new-reproduction-output
```

A new output directory is required for each reproduction; do not use observed test results to tune this benchmark. JSONL split schema follows the scaffold: id, text, label, source_type, template_family, subject_id, conversation_id. The trainer rejects non-synthetic data and checks split leakage before loading the model. It removes only synthetic scenario-number metadata consistently from text.

## Files and safe inference
`output/checkpoint/` is an actual ~17 MB safetensors checkpoint with tokenizer and configuration. Loading is local-only, safetensors-only, without remote code or pickle. `results/` contains the verified aggregate results, history, length audit, split hashes and verification from the original run; a reproduction writes its own output directory. `output/verification.json` checks weight change from pretrained encoder, two independent reloads with identical probe predictions, abstention, empty-input and overlength rejection. Verification does not rerun the test set.

```sh
.venv/bin/python verify_checkpoint.py
.venv/bin/python predict_cpu.py --input-json input.json
```

The input file is `{"text":"..."}`. Text is not echoed or logged. Output contains a research candidate and scores, but the operational flag remains INSUFFICIENT_EVIDENCE with human_review_required=true. Scores are not a safety determination. No internal rationale is generated.

Keep binary weights and `.venv` outside source control. See LICENSE.pretrained, NOTICE.pretrained.txt, and pretrained/README.md for provenance and attribution. `requirements.txt` is the minimal experiment dependency set. Run these commands from optional/cpu-neural. The actual checkpoint is delivered separately and is not in source control.
