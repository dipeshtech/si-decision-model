# Verified upstream boundary

Source inspection, not runtime verification:

- [Kev pinned source](https://github.com/jaredpalmer/kev/tree/1f04cfa28e6a708485b28a9585010b9e436b776e)
- [Training entrypoint](https://github.com/jaredpalmer/kev/blob/1f04cfa28e6a708485b28a9585010b9e436b776e/kev/train.py)
- Base: Qwen/Qwen3.8-27B, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

Kev uses a causal language-model backbone with question/candidate representations and a trained pointer head rather than generating classification prose. This package implements interchange and a tiny abstract gradient smoke, not a replacement implementation of Kev.

Choice row shape:

```json
{"state":"input text","questions":{"severity":{"type":"choice","instructions":"approved rubric","criteria":{"LABEL_A":"approved meaning","LABEL_B":"approved meaning"},"label":"LABEL_A"}},"_meta":{"id":"opaque-example-id","group_id":"opaque-subject-id"}}
```

For independent `noul` questions the label is an actual JSON boolean, not a string; unknown labels are omitted. `score` uses zero-based integer indices and list criteria, so it is deliberately not used for SI category names.

Actual entrypoint is `python -m kev.train`; see `train_plan()` for verified argument names. Explicit `--device cuda --dtype bf16 --weights_dtype bf16` matters: bf16 execution requires CUDA and backbone storage dtype is separately configured. `--max_state` defaults to only 384; supported range is 384–65536. The example plan chooses 2048 pending actual context analysis. These values are placeholders for experimentation, not tuned clinical hyperparameters.

Default upstream mode is LoRA and a new pointer head. Released Kev-27B v2 full weights require matching `--full_ft 1` and compatible checkpoint configuration, not ordinary LoRA warm-start. Compatibility includes base/revision, LoRA/head dimensions, option isolation, special embeddings and weights mode. The 27B reference training setup uses substantial multi-GPU capacity; no costs or GPU capacity are authorized or provisioned by this repository. Never run a remote script blindly or add unpinned model-code execution.

Jev TypeSafe is proprietary. This is Kev-inspired infrastructure and does not claim architectural/training equivalence with Jev, nor does it redistribute either model's weights or upstream source. Review upstream and model licenses before downloading, using, or distributing them.

Upstream Kev requires Python >=3.12,<3.14, while this lightweight standalone package supports Python 3.11+. Use a separate compatible environment for upstream work. Do not install the unrelated PyPI package named `kev`; use only the reviewed official pinned source. No upstream dependencies were installed here.
